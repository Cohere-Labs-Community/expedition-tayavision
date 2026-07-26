"""Alignment pre-training pipeline for Tiny Aya Vision.

Accelerator behaviour (device, autocast, distribution, sampling, stepping) lives
behind `src/backend/`. Selected by TAYAVISION_TPU=1; defaults to DDP + CUDA.

Phase 1 training — trains only the multi-modal projector (connector) to align
vision encoder features with the LLM embedding space, using LLaVA-Pretrain
image-caption pairs.

  - Vision encoder: frozen
  - Multi-modal projector: trainable
  - LLM backbone: frozen

Launch:
  Single GPU:  python pipeline/train_alignment.py
  Multi GPU:   torchrun --nproc_per_node=NUM_GPUS pipeline/train_alignment.py
"""

import json
import os
import sys
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

from config.training_config import AlignmentConfig
from config.model_config import TinyAyaVisionConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from pipeline.data import AlignmentDataset, collate_fn
from pipeline.utils import (
    _unwrap_model,
    save_checkpoint,
    fetch_checkpoint_from_gcs,
    find_latest_checkpoint,
    resolve_resume_run_id,
    build_lr_scheduler,
)
from src.backend import create_backend
from src.processing import TinyAyaVisionProcessor


@torch.no_grad()
def generate_samples(
    model,
    batch: dict,
    processor: TinyAyaVisionProcessor,
    compute_dtype: torch.dtype,
    device: torch.device,
    backend,
    max_new_tokens: int = 256,
    num_samples: int = 2,
) -> list[dict[str, str]]:
    """Generate text from the first `num_samples` items in a batch.

    Pre-computes inputs_embeds with vision features merged so that
    generate() never needs to call _merge_image_features itself.
    """
    raw = _unwrap_model(model)
    was_training = raw.training
    raw.eval()

    results = []
    n = min(num_samples, batch["input_ids"].size(0))
    for i in range(n):
        labels_i = batch["labels"][i]
        response_mask = labels_i != -100
        if response_mask.any():
            prompt_len = response_mask.nonzero(as_tuple=False)[0].item()
        else:
            prompt_len = labels_i.size(0)

        prompt_ids = batch["input_ids"][i, :prompt_len].unsqueeze(0).to(device)
        prompt_mask = batch["attention_mask"][i, :prompt_len].unsqueeze(0).to(device)
        pixel_values = batch["pixel_values"][i].unsqueeze(0).to(device)

        with backend.autocast(compute_dtype):
            inputs_embeds = raw.get_input_embeddings()(prompt_ids)
            image_features = raw.get_image_features(pixel_values)
            inputs_embeds = raw._merge_image_features(
                prompt_ids, inputs_embeds, image_features,
            )

            gen_ids = raw.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_ids = gen_ids[0, 1:]
        prompt_text = processor.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        gen_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Denormalize pixel values → [0, 255] PIL image for wandb
        img_tensor = batch["pixel_values"][i].float().cpu()
        mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
        img_tensor = (img_tensor * std + mean).clamp(0, 1)
        img_pil = Image.fromarray(
            (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

        results.append({
            "image": wandb.Image(img_pil),
            "prompt": prompt_text,
            "generation": gen_text,
        })

    if was_training:
        raw.train()
    return results


def train(
    model,
    dataloader: torch.utils.data.DataLoader,
    sampler: DistributedSampler | None,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_config: AlignmentConfig,
    checkpoint_dir: Path,
    compute_dtype: torch.dtype,
    device: torch.device,
    image_token_id: int,
    processor: TinyAyaVisionProcessor | None = None,
    step_offset: int = 0,
    backend=None,
    batches_to_skip: int = 0,
):
    model.train()
    accumulated_loss = 0.0
    accumulated_ce_loss = 0.0
    accumulated_align_reg_loss = 0.0
    max_image_tokens_in_window = 0
    if backend is None:  # direct callers / tests get the GPU path unchanged
        backend = create_backend("gpu")
    is_main = backend.is_main

    # Accumulate generation samples across save steps
    generation_rows = []

    # Forward hook to capture projector output norms
    norm_cache = {}
    raw = _unwrap_model(model)
    def _projector_hook(module, input, output):
        with torch.no_grad():
            norms = output.detach().float().norm(dim=-1)
            norm_cache["projector_token"] = norms
    hook_handle = raw.multi_modal_projector.register_forward_hook(_projector_hook)

    # Wall time per OPTIMIZER step (i.e. across the whole grad-acc window), which
    # is the unit `playbook/perf-metrics-schema.md` compares across slices.
    import time as _time
    _last_step_t = _time.monotonic()
    _step_is_first = True  # the image-merge guard only needs to fire once

    for epoch in range(training_config.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{training_config.num_epochs}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        # Heartbeats, printed on EVERY host (not just main) and unconditionally.
        # Between them sits the loader; after them sits XLA tracing + compile.
        # Without these two lines a stall in either place looks identical from
        # the log: a v6e-16 run once sat silent for 2h39m between the `[shard]`
        # line and step 1, and the log could not say which half was stuck.
        print(f"[loop] rank={backend.rank} epoch={epoch} awaiting first batch",
              flush=True)
        _first_batch = True
        for _i, batch in enumerate(pbar):
            if _first_batch:
                _first_batch = False
                print(f"[loop] rank={backend.rank} first batch received "
                      f"shape={tuple(batch['input_ids'].shape)}; "
                      "tracing + compiling step 1", flush=True)
            # Skip AFTER the sampler has sharded, so a resume cannot move shard
            # boundaries. Cheap: the collate never runs for skipped batches.
            if _i < batches_to_skip:
                continue
            # `_i` indexes whatever the loader yields, and the two resume
            # strategies yield different things:
            #
            #   skip   (TPU)  -- the loader still yields the WHOLE epoch, so `_i`
            #                    is already the absolute step. Adding the offset
            #                    double-counts it.
            #   subset (GPU)  -- the loader yields only the remainder, so `_i`
            #                    restarts at 0 and the offset is required.
            #
            # Getting this wrong is quiet: data is still correct (the skip
            # itself is right), but the step counter, the LR schedule position,
            # and the `max_steps` test all jump ahead by `resume_step`. Measured
            # on a v6e-16 resume from step 8: the first resumed batch reported
            # step 16 and `max_steps` tripped immediately, so the run did zero
            # further work and still exited "Training complete".
            step = _i if batches_to_skip else step_offset + _i
            if (training_config.max_steps is not None
                    and (step + 1) // training_config.grad_acc_steps
                    >= training_config.max_steps):
                if is_main:
                    print(f"max_steps={training_config.max_steps} reached", flush=True)
                break
            input_ids, attention_mask, pixel_values, labels = (
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                batch["pixel_values"].to(device, non_blocking=True),
                batch["labels"].to(device, non_blocking=True),
            )
            image_grid_hws = batch.get("image_grid_hws")
            if image_grid_hws is not None:
                image_grid_hws = image_grid_hws.to(device, non_blocking=True)

            max_image_tokens_in_window = max(
                max_image_tokens_in_window,
                (input_ids == image_token_id).sum(dim=1).max().item(),
            )

            with backend.autocast(compute_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_hws=image_grid_hws,
                    labels=labels,
                    use_cache=False,
                )
                ce_loss = outputs.loss / training_config.grad_acc_steps
                image_hidden_states = outputs.image_hidden_states  # (B,V,D) or (tokens,D)

                # image_hidden_states is None exactly when the model did NOT merge
                # any image features -- i.e. `pixel_values is None` or the model
                # found none of its image_token_id in input_ids
                # (models/tiny_aya_vision.py:196). That means the batch trained as
                # TEXT ONLY. For an alignment run whose entire purpose is aligning
                # the projector, that is not a degraded run, it is a meaningless
                # one, so fail loudly on the first batch rather than produce a
                # plausible-looking loss curve.
                if image_hidden_states is None and _step_is_first:
                    n_img_tok = int((input_ids == image_token_id).sum())
                    raise RuntimeError(
                        "model returned image_hidden_states=None: no image features "
                        "were merged, so this batch trained on text alone.\n"
                        f"  pixel_values is None      : {pixel_values is None}\n"
                        f"  train() image_token_id    : {image_token_id}\n"
                        f"  model.image_token_id      : {getattr(raw, 'image_token_id', '<unset>')}\n"
                        f"  image tokens in input_ids : {n_img_tok}\n"
                        f"  input_ids shape           : {tuple(input_ids.shape)}\n"
                        "If the two ids differ, setup_tokenizer() did not take effect. "
                        "If the count is 0, the collate truncated them away."
                    )
                _step_is_first = False

                # Only compute the alignment regulariser when it is actually
                # weighted. It defaults to embed_align_reg=0.0, so this was pure
                # wasted compute on every step -- two full-vocab mean/std
                # reductions over a 262144 x 2048 embedding matrix.
                if training_config.embed_align_reg:
                    token_embeddings = raw.language_model.get_input_embeddings().weight
                    # Flatten to 2-D for SigLIP (B,V,D) and MoonViT (tokens,D) alike
                    ihs = image_hidden_states.reshape(-1, image_hidden_states.shape[-1])
                    align_reg_loss = (
                        (token_embeddings.mean(dim=0) - ihs.mean(dim=0)).square().sum()
                        + (token_embeddings.std(dim=0) - ihs.std(dim=0)).square().sum()
                    ) / training_config.grad_acc_steps
                else:
                    align_reg_loss = torch.zeros((), device=ce_loss.device,
                                                 dtype=ce_loss.dtype)

            loss = ce_loss + training_config.embed_align_reg * align_reg_loss
            loss.backward()

            accumulated_loss += loss.item()
            accumulated_ce_loss += ce_loss.item()
            accumulated_align_reg_loss += align_reg_loss.item()

            if (step + 1) % training_config.grad_acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw.multi_modal_projector.parameters(), training_config.max_grad_norm
                )
                backend.optimizer_step(optimizer)
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step = (step + 1) // training_config.grad_acc_steps

                # --- device work: EVERY host, identically --------------------
                #
                # Under SPMD every host runs the same program, and any read of a
                # device tensor back to host (.item(), .cpu(), state_dict()) is
                # effectively a collective: it forces a graph execution that all
                # hosts must join. Putting one behind `if is_main` desynchronises
                # the mesh and deadlocks it -- silently, with no traceback.
                #
                # Measured 2026-07-25 on the v6e-16, tag deploy-20260725-122244:
                # rank 0 sat forever in `pn.std().item()` while ranks 1-3 had
                # already raced into step 2's forward at tiny_aya_vision.py:162.
                # Each group waited on a collective the other would never issue;
                # all four hosts idle, HBM resident, duty cycle 0%.
                #
                # So: compute every scalar here, on all ranks. Only the pure
                # host-side side effects below (wandb, tqdm, disk) are gated.
                _now = _time.monotonic()
                _step_time = _now - _last_step_t
                _last_step_t = _now
                _grad_norm = grad_norm.item()

                log_dict = {
                    "train/loss": accumulated_loss,
                    "train/ce_loss": accumulated_ce_loss,
                    "train/align_reg_loss": accumulated_align_reg_loss,
                    "train/grad_norm": _grad_norm,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/max_image_tokens": max_image_tokens_in_window,
                    # perf/* and mem/* per playbook/perf-metrics-schema.md.
                    # Recorded on BOTH backends so GPU and TPU runs are
                    # comparable on the same axes.
                    "perf/step_time": _step_time,
                    "perf/images_per_sec": (
                        input_ids.size(0) * training_config.grad_acc_steps
                        / max(_step_time, 1e-9)
                    ),
                }
                _mem = backend.memory_info()
                if _mem:
                    log_dict["mem/used_gib"] = _mem.get("used_gib", 0.0)
                    log_dict["mem/total_gib"] = _mem.get("total_gib", 0.0)
                if backend.name == "tpu":
                    # A RISING compile-cause count mid-run means a shape is
                    # varying -- the single most useful XLA signal there is.
                    try:
                        import torch_xla.debug.metrics as _met
                        log_dict["xla/compile_cause_count"] = (
                            _met.metrics_report().count("CompileTime")
                        )
                    except Exception:  # noqa: BLE001 - never break a run to log
                        pass

                if "projector_token" in norm_cache:
                    pn = norm_cache["projector_token"]
                    log_dict["norms/projector_token_mean"] = pn.mean().item()
                    log_dict["norms/projector_token_std"] = pn.std().item()
                    log_dict["norms/projector_token_min"] = pn.min().item()
                    log_dict["norms/projector_token_max"] = pn.max().item()

                    emb_w = raw.get_input_embeddings().weight.detach().float()
                    emb_norms = emb_w.norm(dim=-1)
                    log_dict["norms/emb_matrix_mean"] = emb_norms.mean().item()
                    log_dict["norms/emb_matrix_std"] = emb_norms.std().item()
                    log_dict["norms/emb_matrix_min"] = emb_norms.min().item()
                    log_dict["norms/emb_matrix_max"] = emb_norms.max().item()

                # Checkpointing reads state_dict() off the device, so it is
                # device work too: ALL ranks call it and `backend.save` decides
                # who actually writes the bytes. Gating the call itself would
                # deadlock at the first save -- 500 steps in, hours later.
                if opt_step % training_config.save_steps == 0:
                    save_checkpoint(checkpoint_dir, step + 1, model, optimizer,
                                    lr_scheduler, backend=backend)

                # --- host-side side effects: main only -----------------------
                if is_main:
                    pbar.set_postfix(
                        loss=f"{accumulated_loss:.4f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        gnorm=f"{_grad_norm:.2f}",
                    )

                    if opt_step % training_config.logging_steps == 0:
                        tqdm.write(
                            f"Epoch {epoch}, Opt Step {opt_step}, "
                            f"Loss {accumulated_loss:.4f}, "
                            f"LR {lr_scheduler.get_last_lr()[0]}"
                        )

                    # In-training sample generation is DISABLED on TPU.
                    # `_prepare_cache_for_generation` forces a DynamicCache
                    # whose shape grows by one every decoded token, so on XLA
                    # each token is a new HLO program -- up to 256 compiles
                    # per logged sample. That is tpu-diagnoser row 13, and it
                    # would first fire at save_steps=500, hours in.
                    # Generation belongs off-TPU or behind a static cache.
                    # It stays main-only because it is heavy rank-0 device work
                    # that the TPU path never reaches.
                    if (opt_step % training_config.save_steps == 0
                            and processor is not None and backend.name != "tpu"):
                        samples = generate_samples(
                            model, batch, processor,
                            compute_dtype, device, backend,
                        )
                        for s in samples:
                            generation_rows.append([opt_step, s["image"], s["prompt"], s["generation"]])
                        table = wandb.Table(
                            columns=["step", "image", "prompt", "generation"],
                            data=generation_rows,
                        )
                        log_dict["generations"] = table

                    wandb.log(log_dict, step=opt_step)

                backend.barrier()

                accumulated_loss = 0.0
                accumulated_ce_loss = 0.0
                accumulated_align_reg_loss = 0.0
                max_image_tokens_in_window = 0

    hook_handle.remove()
    # All ranks: same reason as the periodic save above -- reading state_dict()
    # off the device is collective, and `backend.save` picks the writer.
    save_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler,
                    backend=backend)
    backend.barrier()
    if is_main:
        print("Training complete")


def run(cfg: DictConfig):
    """Core training logic. Accelerator behaviour lives behind src/backend/."""
    backend = create_backend()
    backend.setup()
    use_ddp = backend.name == "gpu" and backend.world_size > 1
    device = backend.device
    is_main = backend.is_main

    # TWO divisors, deliberately not the same number:
    #   world_size          = processes/hosts -> how the DataLoader splits
    #   data_parallel_size  = chips           -> how the mesh shards on device
    # On the v6e-16 that is 4 and 16. Conflating them is what made the old
    # `batch_size % world_size` assert fail at 8 % 16.
    world_size = backend.world_size
    n_chips = backend.data_parallel_size

    # 1. Translate omegaconf nodes back to plain dictionaries
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    
    # 2. Re-instantiate your configurations to maintain types downwards
    training_config = AlignmentConfig(**training_dict)

    backend.manual_seed(training_config.seed)
    
    # Instantiate Model Config 
    model_config = TinyAyaVisionConfig.for_encoder(
        cfg.vision.vision_encoder_type, 
        llm=cfg.llm
    )

    # Topology guards. Every failure these catch is otherwise SILENT -- a
    # mis-sharded run hangs or trains on a malformed global batch rather than
    # raising. Fail here, loudly, before a single chip is spent.
    problems = backend.check_topology(global_batch=training_config.batch_size)
    if problems:
        raise ValueError(
            "backend topology check failed:\n  - " + "\n  - ".join(problems)
        )
    per_gpu_batch_size = training_config.batch_size // world_size

    if is_main:
        mode = "DDP" if use_ddp else ("SPMD" if backend.name == "tpu" else "Single-GPU")
        print(f"{mode} [{backend.describe()}]: hosts={world_size} chips={n_chips}, "
              f"global_batch={training_config.batch_size}, "
              f"per_host_batch={per_gpu_batch_size}, "
              f"per_chip={training_config.batch_size // n_chips}")

    # Allow CLI-based resuming (e.g. `python train.py resume=xyz123`)
    resume_run_id = cfg.get("resume", None)
    if not resume_run_id:
        # Fall back to TAYAVISION_RESUME, which the launcher and the QR metadata
        # carry so a recycled node self-heals. An explicit Hydra override always
        # wins: command line beats environment.
        resume_run_id = resolve_resume_run_id()

    if resume_run_id:
        run_id = resume_run_id
    else:
        run_id = str(uuid.uuid4())

    checkpoint_dir = Path(training_config.models_dir) / run_id
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Checkpoint dir: {checkpoint_dir}")
    backend.barrier()

    config_path = checkpoint_dir / "config.json"
    if is_main and not config_path.exists():
        with open(config_path, "w") as f:
            json.dump({
                "training_config": asdict(training_config),
                "model_config": model_config.to_dict(),
            }, f, indent=2)

    if is_main:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=run_id,
            id=run_id.replace("-", ""),
            resume="allow",
            config={
                **asdict(training_config),
                **model_config.to_dict(),
                # Without these a step_time on a chart is unattributable.
                "backend": backend.name,
                "data_parallel_size": backend.data_parallel_size,
                "trc_profile": os.environ.get("TRC_PROFILE", ""),
                "tpu_strategy": os.environ.get("TPU_STRATEGY", ""),
            },
        )

    model = TinyAyaVisionForConditionalGeneration(
        config=model_config,
    )

    processor = TinyAyaVisionProcessor(
        config=model_config,
    )

    model.setup_tokenizer(processor.tokenizer)

    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False

    model.to(device, non_blocking=True)

    compute_dtype = getattr(torch, training_config.torch_dtype)
    model.vision_encoder.to(dtype=compute_dtype, non_blocking=True)
    model.language_model.to(dtype=compute_dtype, non_blocking=True)

    # Gradient checkpointing is what keeps activations to per-layer boundaries,
    # so it stays on for BOTH backends -- it is the difference between ~8.7 GiB
    # and something that will not fit.
    #
    # BOTH torch checkpoint paths call `_get_device_module` ->
    # `getattr(torch, "xla")`, and the non-reentrant one calls it
    # UNCONDITIONALLY -- checkpoint.py:1476 runs twelve lines before the
    # `if preserve_rng_state:` guard at :1488. So `preserve_rng_state=False`
    # does NOT avoid it, and neither does `DefaultDeviceType.set_device_type()`.
    # What makes this work is the `torch.xla` shim that `TPUBackend.setup()`
    # installs (see `src/backend/tpu_backend.py::_register_xla_device_module`,
    # called from `backend.setup()` above, long before this line).
    #
    # preserve_rng_state stays TRUE on XLA, and that is deliberate. The flag
    # governs two independent things and only one of them involves the shim:
    #
    #   * device RNG -- gated on `getattr(device_module, "_initialized", False)`
    #     (checkpoint.py:1496). The shim deliberately omits `_initialized`, so
    #     this is always skipped, `had_device_in_fwd` stays False, `rng_devices`
    #     stays `[]`, and `get_device_states` is never reached. Verified: with
    #     `devices=[]`, `fork_rng` touches no attribute on the shim at all.
    #   * CPU RNG -- `torch.get_rng_state()` / `torch.set_rng_state()` at
    #     checkpoint.py:1489 and :1511. NOT gated on `_initialized`.
    #
    # So True costs nothing on XLA, keeps the CPU generator consistent between
    # forward and recompute, and matches train_instruct.py / train_multilingual
    # which both leave it at its default of True. Setting it False bought no
    # memory and no speed -- it only made this branch silently diverge.
    if backend.name == "tpu":
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.language_model.gradient_checkpointing_enable()

    model = backend.wrap_model(model)
    # torch.compile is a CUDA/inductor path. On XLA the graph is already
    # traced+compiled by the runtime, and wrapping it here causes a second,
    # conflicting compilation layer.
    if backend.name == "gpu":
        model = torch.compile(model)

    resume_step = 0
    if resume_run_id:
        # A recycled slice comes back with an empty /models, so consult the
        # gs:// mirror before concluding there is nothing to resume from --
        # otherwise a preemption silently restarts training at step 0 while
        # every log line still says "resume". No-op when SAVE_CKPT_DIR is unset
        # or a local checkpoint is already present.
        #
        # Every rank fetches: each host has its own /models and must load the
        # weights into its own replica.
        fetch_checkpoint_from_gcs(checkpoint_dir)
        backend.barrier()

        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            if is_main:
                print(f"Resuming from {ckpt_path}")
            # torch has no XLA deserializer, so mapping straight onto the
            # device raises "don't know how to restore data location of
            # torch.storage.UntypedStorage (tagged with xla:0)" -- note the tag
            # is the TARGET, not the file. Load to host and let
            # `load_state_dict` copy into the already-placed parameters.
            # GPU keeps `map_location=device` so the Modal path is unchanged.
            map_location = "cpu" if backend.name == "tpu" else device
            ckpt = torch.load(ckpt_path, map_location=map_location)
            raw_model = _unwrap_model(model)
            raw_model.multi_modal_projector.load_state_dict(ckpt["projector"])
            resume_step = ckpt["step"]
            if is_main:
                print(f"Resuming from step {resume_step}")
        else:
            if is_main:
                print(f"No checkpoints found in {checkpoint_dir}, starting from scratch")

    dataset = AlignmentDataset(
        config=model_config,
        dataset_name=training_config.dataset_name,
        data_dir=training_config.data_dir,
    )

    full_dataset_len = len(dataset)

    # SHARD-THEN-SKIP on TPU, never skip-then-shard.
    #
    # Subsetting first changes the dataset the sampler shards, so shard
    # boundaries MOVE across a resume and hosts silently swap data. On spot
    # capacity with qr_watch.sh recycling, resumes are routine.
    #
    # The GPU path keeps its original skip-then-shard order deliberately: it is
    # the behaviour every published checkpoint was produced with, and changing
    # it here would make this commit non-equivalent. It has the same latent
    # issue under multi-GPU DDP resume -- tracked, not silently fixed.
    batches_to_skip = 0
    samples_to_skip = resume_step * per_gpu_batch_size
    if backend.name == "tpu":
        if resume_step > 0:
            batches_to_skip = resume_step
            if is_main:
                print(f"Resume: sharding the FULL dataset, then skipping "
                      f"{batches_to_skip} batches per host")
    elif samples_to_skip > 0 and samples_to_skip < len(dataset):
        remaining_indices = list(range(samples_to_skip, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, remaining_indices)
        if is_main:
            print(f"Skipped {samples_to_skip} samples, {len(dataset)} remaining")

    # Multi-host: shards across processes. Single-host: None.
    sampler = backend.make_sampler(dataset, shuffle=True, seed=training_config.seed)

    if sampler is not None:
        print(f"[shard] process_index={backend.rank}/{backend.world_size} "
              f"local_shard={len(sampler)} of dataset={len(dataset)}", flush=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            pad_token_id=processor.tokenizer.pad_token_id,
            # XLA needs ONE sequence length or it recompiles per batch shape.
            # None on GPU keeps dynamic padding, i.e. Modal is unchanged.
            fixed_seq_len=(training_config.fixed_seq_len
                           if backend.name == "tpu" else None),
            # Lets the collate refuse to truncate image tokens away.
            image_token_id=processor.image_token_id,
        ),
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=training_config.num_workers > 0,
        prefetch_factor=2 if training_config.num_workers > 0 else None,
        # TPU: the ragged final batch is fatal, not merely wasteful. The sampler
        # already drops its own tail (139,532 per host), but 139,532/64 leaves a
        # final batch of 12 -- and 12 is not divisible by the 4 local chips, so
        # ShardingSpec(minibatch=True) raises on it. It is also a new shape, i.e.
        # an XLA recompile at every epoch boundary. Costs 12 samples per host.
        # GPU keeps False so the Modal path is byte-identical.
        drop_last=(backend.name == "tpu"),
    )

    raw_model = _unwrap_model(model)
    opt = torch.optim.AdamW(
        raw_model.multi_modal_projector.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    lr_scheduler = build_lr_scheduler(opt, training_config, full_dataset_len, per_gpu_batch_size, world_size)

    if resume_step > 0:
        opt.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    train(
        model=model,
        # MUST go through the backend: on TPU this is what shards the batch
        # across chips (MpDeviceLoader input_sharding). Identity on GPU.
        dataloader=backend.wrap_loader(loader),
        sampler=sampler,
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        compute_dtype=compute_dtype,
        device=device,
        image_token_id=processor.image_token_id,
        processor=processor,
        step_offset=resume_step,
        backend=backend,
        batches_to_skip=batches_to_skip,
    )

    if is_main:
        wandb.finish()
    backend.cleanup()


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
