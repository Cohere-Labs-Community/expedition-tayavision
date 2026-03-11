"""Tests for LoRA adapter configuration and optimizer group utilities.

All tests here are CPU-only and do not load the full 3.35B LLM.
The apply_lora() integration path (which downloads ~6.7 GB) is covered by
test_vlm_assembly.py and is skipped when CUDA is unavailable.
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType

from config.lora_config import LoraAdapterConfig
from scripts.apply_lora import count_parameters, get_lora_optimizer_groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLoraModel(nn.Module):
    """Tiny model whose parameter names mimic a PEFT-wrapped module.

    Lets us test get_lora_optimizer_groups() without loading any real weights.
    """

    def __init__(self):
        super().__init__()
        # Adapter matrices (named to match PEFT's convention)
        self.layer0_lora_A_weight = nn.Parameter(torch.randn(8, 4))
        self.layer0_lora_B_weight = nn.Parameter(torch.randn(4, 8))
        self.layer1_lora_A_weight = nn.Parameter(torch.randn(8, 4))
        self.layer1_lora_B_weight = nn.Parameter(torch.randn(4, 8))
        # Other trainable params (e.g. connector)
        self.connector = nn.Linear(8, 8)
        # Frozen param (requires_grad=False)
        self.frozen_weight = nn.Parameter(torch.randn(8, 8), requires_grad=False)

    def forward(self, x):  # pragma: no cover
        return x


# ---------------------------------------------------------------------------
# LoraAdapterConfig
# ---------------------------------------------------------------------------


class TestLoraAdapterConfig:
    def test_default_rank_and_alpha(self):
        cfg = LoraAdapterConfig()
        assert cfg.rank == 256
        assert cfg.lora_alpha == 512

    def test_default_scale(self):
        """Alpha/rank scaling should be 2.0 by default."""
        cfg = LoraAdapterConfig()
        assert cfg.lora_alpha / cfg.rank == 2.0

    def test_default_dropout(self):
        cfg = LoraAdapterConfig()
        assert cfg.lora_dropout == 0.05

    def test_mid_to_top_layers(self):
        """Should target the upper 18 layers (18–35) of the 36-layer LLM."""
        cfg = LoraAdapterConfig()
        assert cfg.layers_to_transform == list(range(18, 36))
        assert len(cfg.layers_to_transform) == 18

    def test_lower_layers_excluded(self):
        """Layers 0–17 (lower half) must not be in the default list."""
        cfg = LoraAdapterConfig()
        for idx in range(18):
            assert idx not in cfg.layers_to_transform

    def test_attention_modules_targeted(self):
        cfg = LoraAdapterConfig()
        for mod in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert mod in cfg.target_modules, f"{mod} missing from target_modules"

    def test_mlp_modules_targeted(self):
        cfg = LoraAdapterConfig()
        for mod in ("gate_proj", "up_proj", "down_proj"):
            assert mod in cfg.target_modules, f"{mod} missing from target_modules"

    def test_uniform_lr_multipliers_by_default(self):
        cfg = LoraAdapterConfig()
        assert cfg.lora_a_lr_multiplier == 1.0
        assert cfg.lora_b_lr_multiplier == 1.0

    def test_custom_rank_and_alpha(self):
        cfg = LoraAdapterConfig(rank=64, lora_alpha=128)
        assert cfg.rank == 64
        assert cfg.lora_alpha == 128

    def test_custom_layer_range(self):
        cfg = LoraAdapterConfig(layers_to_transform=list(range(24, 36)))
        assert cfg.layers_to_transform == list(range(24, 36))


class TestLoraAdapterConfigToPeftConfig:
    def test_returns_lora_config_instance(self):
        peft_cfg = LoraAdapterConfig().to_peft_config()
        assert isinstance(peft_cfg, LoraConfig)

    def test_task_type_is_causal_lm(self):
        peft_cfg = LoraAdapterConfig().to_peft_config()
        assert peft_cfg.task_type == TaskType.CAUSAL_LM

    def test_rank_propagated(self):
        cfg = LoraAdapterConfig(rank=128)
        peft_cfg = cfg.to_peft_config()
        assert peft_cfg.r == 128

    def test_alpha_propagated(self):
        cfg = LoraAdapterConfig(lora_alpha=256)
        peft_cfg = cfg.to_peft_config()
        assert peft_cfg.lora_alpha == 256

    def test_dropout_propagated(self):
        cfg = LoraAdapterConfig(lora_dropout=0.1)
        peft_cfg = cfg.to_peft_config()
        assert peft_cfg.lora_dropout == 0.1

    def test_layers_to_transform_propagated(self):
        layers = list(range(20, 36))
        peft_cfg = LoraAdapterConfig(layers_to_transform=layers).to_peft_config()
        assert peft_cfg.layers_to_transform == layers

    def test_target_modules_propagated(self):
        mods = ["q_proj", "v_proj"]
        peft_cfg = LoraAdapterConfig(target_modules=mods).to_peft_config()
        # PEFT may store as set or list; check membership
        for m in mods:
            assert m in peft_cfg.target_modules


# ---------------------------------------------------------------------------
# get_lora_optimizer_groups
# ---------------------------------------------------------------------------


class TestGetLoraOptimizerGroups:
    @pytest.fixture
    def model(self):
        return _FakeLoraModel()

    @pytest.fixture
    def lora_config(self):
        return LoraAdapterConfig()

    def test_returns_three_groups(self, model, lora_config):
        groups = get_lora_optimizer_groups(model, base_lr=1e-4, lora_config=lora_config)
        assert len(groups) == 3

    def test_group_names(self, model, lora_config):
        groups = get_lora_optimizer_groups(model, base_lr=1e-4, lora_config=lora_config)
        names = {g["name"] for g in groups}
        assert names == {"lora_A", "lora_B", "other"}

    def test_frozen_params_excluded(self, model, lora_config):
        groups = get_lora_optimizer_groups(model, base_lr=1e-4, lora_config=lora_config)
        grouped_ids = {id(p) for g in groups for p in g["params"]}
        assert id(model.frozen_weight) not in grouped_ids

    def test_all_trainable_params_covered(self, model, lora_config):
        groups = get_lora_optimizer_groups(model, base_lr=1e-4, lora_config=lora_config)
        grouped = {id(p) for g in groups for p in g["params"]}
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        assert grouped == expected

    def test_uniform_lr_when_multipliers_are_one(self, model, lora_config):
        base_lr = 2e-4
        groups = get_lora_optimizer_groups(model, base_lr=base_lr, lora_config=lora_config)
        for g in groups:
            assert g["lr"] == pytest.approx(base_lr)

    def test_differential_lr_lora_a(self, model):
        cfg = LoraAdapterConfig(lora_a_lr_multiplier=0.5, lora_b_lr_multiplier=1.0)
        base_lr = 1e-4
        groups = get_lora_optimizer_groups(model, base_lr=base_lr, lora_config=cfg)
        group = next(g for g in groups if g["name"] == "lora_A")
        assert group["lr"] == pytest.approx(base_lr * 0.5)

    def test_differential_lr_lora_b(self, model):
        cfg = LoraAdapterConfig(lora_a_lr_multiplier=1.0, lora_b_lr_multiplier=2.0)
        base_lr = 1e-4
        groups = get_lora_optimizer_groups(model, base_lr=base_lr, lora_config=cfg)
        group = next(g for g in groups if g["name"] == "lora_B")
        assert group["lr"] == pytest.approx(base_lr * 2.0)

    def test_other_group_always_uses_base_lr(self, model):
        cfg = LoraAdapterConfig(lora_a_lr_multiplier=0.1, lora_b_lr_multiplier=10.0)
        base_lr = 3e-4
        groups = get_lora_optimizer_groups(model, base_lr=base_lr, lora_config=cfg)
        group = next(g for g in groups if g["name"] == "other")
        assert group["lr"] == pytest.approx(base_lr)

    def test_lora_a_params_correctly_classified(self, model, lora_config):
        groups = get_lora_optimizer_groups(model, base_lr=1e-4, lora_config=lora_config)
        lora_a_group = next(g for g in groups if g["name"] == "lora_A")
        expected = [
            model.layer0_lora_A_weight,
            model.layer1_lora_A_weight,
        ]
        assert len(lora_a_group["params"]) == len(expected)
        for p in expected:
            assert any(p is q for q in lora_a_group["params"])

    def test_lora_b_params_correctly_classified(self, model, lora_config):
        groups = get_lora_optimizer_groups(model, base_lr=1e-4, lora_config=lora_config)
        lora_b_group = next(g for g in groups if g["name"] == "lora_B")
        expected = [
            model.layer0_lora_B_weight,
            model.layer1_lora_B_weight,
        ]
        assert len(lora_b_group["params"]) == len(expected)
        for p in expected:
            assert any(p is q for q in lora_b_group["params"])


# ---------------------------------------------------------------------------
# count_parameters utility
# ---------------------------------------------------------------------------


class TestCountParameters:
    def test_all_trainable(self):
        m = nn.Linear(4, 4)
        trainable, total = count_parameters(m)
        assert trainable == total
        assert trainable == 4 * 4 + 4  # weight + bias

    def test_partially_frozen(self):
        m = nn.Linear(4, 4)
        m.bias.requires_grad = False
        trainable, total = count_parameters(m)
        assert total == 4 * 4 + 4
        assert trainable == 4 * 4  # only weight

    def test_fully_frozen(self):
        m = nn.Linear(4, 4)
        for p in m.parameters():
            p.requires_grad = False
        trainable, total = count_parameters(m)
        assert trainable == 0
        assert total == 4 * 4 + 4
