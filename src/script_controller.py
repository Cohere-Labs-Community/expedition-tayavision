"""Script-Conditioned Token Allocation Controller (SCTAC).

The controller predicts a per-image compression-ratio distribution from
(a) the query language's script-fertility signal and (b) image-side
statistics. It is the mechanism described in the
``Script-Conditioned Adaptive Visual Token Allocation`` proposal.

Phase-1 contract (this implementation):
- Always emit 196 tokens to the connector (same as today's fixed-rate path).
- Produce a soft compression distribution ``p in R^3`` over
  ``compression_levels`` (default [2, 4, 8]) per image.
- Surface ``expected_tokens = sum(p_r * N_r)`` and ``rate_loss`` to the
  training loop so the controller can be regularised toward a token budget.
- When ``use_gumbel=True``: also hard-pick a compression level per image,
  mean-pool the pre-projector features accordingly, and write the pooled
  features into the first ``N_kept`` of the 196 token slots — the rest
  are zeroed and reported via ``keep_mask``. The LLM still sees 196
  ``<image>`` placeholders; masked ones receive zero embeddings.

When ``ScriptControllerConfig.enabled == False``, the module is not
instantiated — see ``TinyAyaVisionForConditionalGeneration``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ScriptControllerConfig:
    """Configuration for the Script-Conditioned Token Allocation Controller."""

    enabled: bool = False
    variant: str = "B1"  # "B1" | "B2" | "B3"
    target_tokens: int = 196
    rate_lambda: float = 0.1
    fairness_lambda: float = 0.0
    use_gumbel: bool = False
    gumbel_tau: float = 1.0
    fertility_table_path: str = ""
    compression_levels: tuple[int, ...] = (2, 4, 8)
    fairness_eval_steps: int = 500

    @classmethod
    def from_dict(cls, d: dict | None) -> "ScriptControllerConfig":
        if not d:
            return cls()
        levels = d.get("compression_levels")
        if levels is not None:
            d = {**d, "compression_levels": tuple(levels)}
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "variant": self.variant,
            "target_tokens": self.target_tokens,
            "rate_lambda": self.rate_lambda,
            "fairness_lambda": self.fairness_lambda,
            "use_gumbel": self.use_gumbel,
            "gumbel_tau": self.gumbel_tau,
            "fertility_table_path": self.fertility_table_path,
            "compression_levels": list(self.compression_levels),
            "fairness_eval_steps": self.fairness_eval_steps,
        }


@dataclass
class FertilityTable:
    """Per-language and per-script tokenizer-fertility lookup.

    ``per_language`` maps a language code (FLORES-200 style ``eng_Latn``)
    to ``{"script": "Latn", "tpw": float, "tpc": float}``.
    ``per_script`` maps an ISO 15924 script code to a mean ``tpw`` value.
    """

    backbone: str
    per_language: dict[str, dict]
    per_script: dict[str, float]

    @classmethod
    def empty(cls) -> "FertilityTable":
        return cls(backbone="", per_language={}, per_script={})

    @classmethod
    def load(cls, path: str | Path) -> "FertilityTable":
        path = Path(path)
        if not path.exists():
            return cls.empty()
        with open(path) as f:
            raw = json.load(f)
        return cls(
            backbone=raw.get("backbone", ""),
            per_language=raw.get("per_language", {}),
            per_script=raw.get("per_script", {}),
        )

    @property
    def scripts(self) -> list[str]:
        return sorted(self.per_script.keys())

    def lookup_tpw(self, lang_code: str | None) -> float:
        """Return tokens-per-word for ``lang_code``.

        Falls back to per-script mean when the exact language is missing,
        then to a uniform 1.0 when the table is empty.
        """
        if not lang_code or not self.per_language:
            return 1.0
        if lang_code in self.per_language:
            return float(self.per_language[lang_code]["tpw"])
        # Try splitting FLORES-style "eng_Latn" → script "Latn"
        if "_" in lang_code:
            script = lang_code.split("_", 1)[1]
            if script in self.per_script:
                return float(self.per_script[script])
        return 1.0

    def lookup_script_index(self, lang_code: str | None) -> int:
        """Return the index of the language's script in ``self.scripts``.

        Returns ``-1`` when no match is found, which the controller treats
        as a fully-zero one-hot.
        """
        if not lang_code or not self.per_script:
            return -1
        if lang_code in self.per_language:
            script = self.per_language[lang_code].get("script", "")
        elif "_" in lang_code:
            script = lang_code.split("_", 1)[1]
        else:
            return -1
        try:
            return self.scripts.index(script)
        except ValueError:
            return -1


class ScriptController(nn.Module):
    """Lightweight MLP that scores per-image compression budgets.

    Architecture (B1/B2): ``concat([s_lang, c_img]) -> Linear(in, 64) -> SiLU
    -> Linear(64, len(compression_levels))``. Total parameters < 100k.

    Forward signature::

        controller(patch_tokens, lang_codes) -> (compressed_tokens, keep_mask, aux)

    where ``compressed_tokens`` is the (post-pixel-shuffle, possibly pooled)
    feature tensor passed to the connector and ``aux`` carries the
    ``expected_tokens`` and ``rate_loss`` scalars used by the training loop.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        cfg: ScriptControllerConfig,
        fertility_table: FertilityTable | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.vision_hidden_size = vision_hidden_size
        self.compression_levels = tuple(cfg.compression_levels)
        self.num_levels = len(self.compression_levels)

        self.fertility_table = fertility_table or FertilityTable.empty()
        self.num_scripts = max(1, len(self.fertility_table.scripts))

        # Input features per variant:
        #   B1: [tpw, script_onehot]                            (1 + num_scripts)
        #   B2: [tpw, script_onehot, c_img]                     (1 + num_scripts + 1)
        #   B3: [tpw, script_onehot, c_img, d_img]              (1 + num_scripts + 2)
        in_dim = 1 + self.num_scripts
        if cfg.variant in ("B2", "B3"):
            in_dim += 1
        if cfg.variant == "B3":
            in_dim += 1
        self.in_dim = in_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.num_levels),
        )

        # Tokens-per-image at each compression level for a 27x27 (padded to 28x28)
        # SigLIP grid pixel-shuffled at r=2: 14x14 = 196 tokens.
        # Subsequent mean-pool by factor (r/2) yields:
        #   r=2 -> 196   (14x14)
        #   r=4 -> 49    (7x7)
        #   r=8 -> 16    (4x4, after padding 14->16)
        base_tokens = 196
        self.register_buffer(
            "tokens_at_level",
            torch.tensor(
                [base_tokens // ((r // 2) ** 2) for r in self.compression_levels],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Input featurisation
    # ------------------------------------------------------------------

    def _build_lang_features(
        self, lang_codes: list[str] | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return (B, 1 + num_scripts) — [tpw, one-hot script]."""
        feats = torch.zeros(batch_size, 1 + self.num_scripts, device=device)
        if not lang_codes:
            feats[:, 0] = 1.0  # neutral fertility
            return feats
        for i, lang in enumerate(lang_codes[:batch_size]):
            tpw = self.fertility_table.lookup_tpw(lang)
            idx = self.fertility_table.lookup_script_index(lang)
            feats[i, 0] = tpw
            if 0 <= idx < self.num_scripts:
                feats[i, 1 + idx] = 1.0
        return feats

    @staticmethod
    def _complexity_scalar(patch_tokens: torch.Tensor) -> torch.Tensor:
        """Image-complexity proxy: variance of patch-token L2 norms.

        Following Adaptive-VoCo, but reduced to a single scalar per image.
        Shape: ``(B,) -> (B, 1)``.
        """
        norms = patch_tokens.norm(dim=-1)  # (B, S)
        return norms.var(dim=1, unbiased=False).unsqueeze(-1)  # (B, 1)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def _pool_to_level(
        self, post_shuffle: torch.Tensor, level: int
    ) -> tuple[torch.Tensor, int]:
        """Mean-pool ``post_shuffle`` further from r=2 down to the requested level.

        ``post_shuffle`` is ``(B, 196, D)`` reshaped from a 14x14 grid.
        Returns ``(pooled_BHW_D, n_tokens)``.
        """
        b, n, d = post_shuffle.shape
        assert n == 196, f"expected 196 tokens at r=2, got {n}"
        if level == 2:
            return post_shuffle, n

        grid = post_shuffle.view(b, 14, 14, d).permute(0, 3, 1, 2)  # (B, D, 14, 14)
        if level == 4:
            pooled = F.avg_pool2d(grid, kernel_size=2, stride=2)  # (B, D, 7, 7)
        elif level == 8:
            # 14 doesn't divide by 4; pad to 16 first.
            grid = F.pad(grid, (1, 1, 1, 1))  # (B, D, 16, 16)
            pooled = F.avg_pool2d(grid, kernel_size=4, stride=4)  # (B, D, 4, 4)
        else:
            raise ValueError(f"unsupported compression level {level}")
        pooled = pooled.permute(0, 2, 3, 1).reshape(b, -1, d)
        return pooled, pooled.shape[1]

    def _apply_hard_compression(
        self,
        post_shuffle: torch.Tensor,
        chosen_level: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample hard-pick compression with zero-padding back to 196 slots.

        Each row of the output has its first ``N_r`` slots filled with the
        pooled features for that sample's chosen level; remaining slots are
        zero. ``keep_mask`` marks the valid (non-padding) slots.
        """
        b, n, d = post_shuffle.shape
        out = torch.zeros_like(post_shuffle)
        keep_mask = torch.zeros(b, n, dtype=torch.bool, device=post_shuffle.device)
        for i in range(b):
            level = int(chosen_level[i].item())
            pooled, n_kept = self._pool_to_level(post_shuffle[i : i + 1], level)
            out[i, :n_kept] = pooled[0]
            keep_mask[i, :n_kept] = True
        return out, keep_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        patch_tokens: torch.Tensor,
        post_shuffle: torch.Tensor,
        lang_codes: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Score compression budgets and (optionally) hard-compress features.

        Args:
            patch_tokens: ``(B, 729, vision_hidden_size)`` — pre-pixel-shuffle
                SigLIP outputs. Used for the image-complexity scalar.
            post_shuffle: ``(B, 196, ps_dim)`` — output of the existing
                pixel-shuffle path. Passed through unchanged in soft mode;
                pooled + padded in Gumbel mode.
            lang_codes: per-sample language codes (FLORES-200 style preferred).

        Returns:
            ``(features_for_connector, keep_mask, aux)`` where
            ``features_for_connector`` matches ``post_shuffle.shape`` and
            ``aux`` contains
            ``{"compression_probs", "expected_tokens", "rate_loss"}``.
        """
        b = patch_tokens.shape[0]
        device = patch_tokens.device

        lang_feats = self._build_lang_features(lang_codes, b, device).to(
            dtype=patch_tokens.dtype
        )
        inputs = [lang_feats]
        if self.cfg.variant in ("B2", "B3"):
            inputs.append(self._complexity_scalar(patch_tokens).to(patch_tokens.dtype))
        if self.cfg.variant == "B3":
            # d_img (image-script density) — placeholder zero, populated
            # when the OCR-density head is wired in a future iteration.
            inputs.append(torch.zeros(b, 1, device=device, dtype=patch_tokens.dtype))

        x = torch.cat(inputs, dim=-1)
        logits = self.mlp(x.float())  # (B, num_levels)

        if self.cfg.use_gumbel and self.training:
            probs = F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=False)
        else:
            probs = F.softmax(logits, dim=-1)

        tokens_at_level = self.tokens_at_level.to(device)
        expected_tokens = (probs * tokens_at_level.unsqueeze(0)).sum(dim=-1)  # (B,)
        rate_loss = (expected_tokens - float(self.cfg.target_tokens)).abs().mean()

        aux = {
            "compression_probs": probs.detach(),
            "expected_tokens": expected_tokens.detach(),
            "rate_loss": rate_loss,
        }

        if self.cfg.use_gumbel:
            chosen = probs.argmax(dim=-1)  # (B,)
            chosen_level = torch.tensor(
                [self.compression_levels[i] for i in chosen.tolist()],
                device=device,
            )
            features, keep_mask = self._apply_hard_compression(post_shuffle, chosen_level)
            aux["chosen_level"] = chosen_level
        else:
            features = post_shuffle
            keep_mask = torch.ones(
                b, post_shuffle.shape[1], dtype=torch.bool, device=device
            )

        return features, keep_mask, aux
