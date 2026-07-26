"""Common utility functions for training pipelines."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from models import save_for_inference
from src.processing import TinyAyaVisionProcessor

# Generous: a ~140 MB projector+optimizer checkpoint to a co-located bucket
# takes seconds. The timeout exists so a wedged network cannot stall training
# indefinitely, not to bound normal operation.
GCS_TIMEOUT_S = 600


def gcs_checkpoint_root() -> str | None:
    """The `gs://` prefix checkpoints mirror to, or None when not configured.

    Reads `SAVE_CKPT_DIR`. Until this function existed that variable reached
    `scripts/tpu/*.sh` -- deploy warnings, queued-resource metadata, the
    `qr_watch.sh` durability gate -- but NOT the training code. Every surface
    therefore reported durability as armed while `save_checkpoint` wrote to
    node-local `/models`, which a preemption destroys. Anything that is not a
    `gs://` URI is ignored, so the Modal/GPU path is unaffected.
    """
    root = os.environ.get("SAVE_CKPT_DIR", "").strip().rstrip("/")
    return root if root.startswith("gs://") else None


def _gcloud_storage(*args: str) -> tuple[bool, str, str]:
    """Run `gcloud storage <args>`. Returns (ok, stdout, error_message).

    Shelling out rather than adding `google-cloud-storage`: every script that
    sets `SAVE_CKPT_DIR` already requires the gcloud CLI, so this introduces no
    dependency that the path did not already have.
    """
    exe = shutil.which("gcloud")
    if exe is None:
        return False, "", "gcloud not on PATH"
    try:
        proc = subprocess.run(
            [exe, "storage", *args],
            capture_output=True, text=True, timeout=GCS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"timed out after {GCS_TIMEOUT_S}s"
    except OSError as exc:
        return False, "", str(exc)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        return False, "", msg.splitlines()[-1] if msg else f"exit {proc.returncode}"
    return True, proc.stdout, ""


def _checkpoint_step(name: str) -> int:
    """Step number encoded in a checkpoint filename, or -1."""
    match = re.search(r"checkpoint_(\d+)\.pt$", str(name))
    return int(match.group(1)) if match else -1


def mirror_checkpoint_to_gcs(local_path: Path) -> bool:
    """Copy a just-written checkpoint to `SAVE_CKPT_DIR/<run_id>/`.

    Call from the rank that actually wrote the file, and only that rank. This
    is safe despite the SPMD rule that device work must not be rank-gated:
    uploading a file that already exists on disk is pure host I/O and issues no
    collective. It does stall the mesh for the upload -- the other ranks run
    ahead and block at the next collective until this one arrives -- which at
    ~seconds per `save_steps` interval is a rounding error against the run.
    """
    root = gcs_checkpoint_root()
    if root is None:
        return False
    # /models/<run_id>/checkpoint_N.pt -> <root>/<run_id>/checkpoint_N.pt,
    # matching the layout the interim rsync loop already produced.
    dest = f"{root}/{local_path.parent.name}/{local_path.name}"
    ok, _, err = _gcloud_storage("cp", str(local_path), dest)
    if ok:
        print(f"Mirrored checkpoint to {dest}", flush=True)
        return True
    # Deliberately does NOT raise. Killing a multi-hour run over a transient
    # bucket error is worse than losing one mirror. But it is loud, because a
    # silent failure here means the run only *looks* durable.
    print(
        f"WARNING: checkpoint NOT mirrored to {dest} ({err}). "
        f"{local_path} exists only on node-local disk and will not survive "
        "a preemption.",
        flush=True,
    )
    return False


# Values that mean "do not resume". Anything else is either `auto` or a literal
# run id, so a typo resolves to a run id that does not exist and reports
# "no mirrored checkpoints" rather than silently starting over.
_RESUME_OFF = frozenset({"", "off", "none", "no", "0", "false"})


def latest_mirrored_run_id() -> str | None:
    """Run id of the most recently mirrored checkpoint under `SAVE_CKPT_DIR`.

    Deterministic across hosts: it is a pure function of the bucket listing, and
    at startup no run is writing to it. Should it ever disagree between hosts,
    `fetch_checkpoint_from_gcs` logs the exact object it pulled on every rank,
    so the divergence is greppable rather than silent.
    """
    root = gcs_checkpoint_root()
    if root is None:
        print("TAYAVISION_RESUME=auto but SAVE_CKPT_DIR is not a gs:// URI -- "
              "nothing durable to resume from. Starting fresh.", flush=True)
        return None
    ok, listing, err = _gcloud_storage("ls", "-l", f"{root}/*/checkpoint_*.pt")
    newest_ts, newest_url = "", None
    if ok:
        for line in listing.splitlines():
            # "<size>  <ISO-8601>  gs://<root>/<run_id>/checkpoint_<n>.pt".
            # ISO-8601 sorts lexicographically, so no date parsing is needed.
            # The trailing "TOTAL: ..." summary has no gs:// field and is skipped.
            fields = line.split()
            if len(fields) < 3 or not fields[-1].startswith("gs://"):
                continue
            if fields[-2] > newest_ts:
                newest_ts, newest_url = fields[-2], fields[-1]
    if newest_url is None:
        print(f"TAYAVISION_RESUME=auto found no checkpoints under {root} "
              f"({err or 'empty'}). Starting fresh.", flush=True)
        return None
    run_id = newest_url.rstrip("/").split("/")[-2]
    print(f"TAYAVISION_RESUME=auto -> run {run_id} "
          f"(newest mirrored checkpoint, {newest_ts})", flush=True)
    return run_id


def resolve_resume_run_id() -> str | None:
    """Translate `TAYAVISION_RESUME` into a run id, or None to start fresh.

    `off`/`none`/`no`/`0`/`false`/unset -> None
    `auto`                              -> newest run id under SAVE_CKPT_DIR
    anything else                       -> taken as a literal run id

    Until this existed the variable was inert: `train_launcher.sh:93` printed
    `resume=${TAYAVISION_RESUME:-auto}` and nothing read it, so every surface --
    the launcher banner, the QR metadata, `SPEC.md` section 5 -- described a
    self-healing resume that could not happen. It is the same shape of defect
    `SAVE_CKPT_DIR` had: a knob that reads as wired and is not.

    Callers must let an explicit Hydra `resume=<uuid>` win over this, because
    the command line beating the environment is this repo's config convention.

    **`auto` is only safe where the config is fixed.** It resumes whatever ran
    last, so on the boot/recycle path -- same QR metadata, therefore same
    overrides by construction -- it is exactly right. On a human redeploy, where
    the overrides change between invocations, it would load run A's optimizer
    state and step count into run B's config, silently. That is why the default
    is set per caller (`launch_qr.sh`/`startup_script.sh` -> auto,
    `deploy_tarball.sh` -> off) rather than once in `_lib.sh`.
    """
    raw = os.environ.get("TAYAVISION_RESUME", "").strip()
    if raw.lower() in _RESUME_OFF:
        return None
    if raw.lower() != "auto":
        return raw
    return latest_mirrored_run_id()


def fetch_checkpoint_from_gcs(checkpoint_dir: Path) -> Path | None:
    """Pull the newest mirrored checkpoint for this run into `checkpoint_dir`.

    Call on EVERY rank. Each host has its own `/models` and each must load the
    same weights into its own replica; fetching on rank 0 alone would leave the
    others with a freshly-initialised projector -- divergent state, and silent.

    A local checkpoint always wins, so this is a no-op on a slice that never
    went away. It only does work after a recycle, which is exactly the case
    `find_latest_checkpoint` would otherwise report as "start from scratch".
    """
    root = gcs_checkpoint_root()
    if root is None or find_latest_checkpoint(checkpoint_dir) is not None:
        return None
    prefix = f"{root}/{checkpoint_dir.name}"
    ok, listing, err = _gcloud_storage("ls", f"{prefix}/checkpoint_*.pt")
    remote = [u for u in listing.split() if u.endswith(".pt")] if ok else []
    if not remote:
        print(f"No mirrored checkpoints under {prefix} ({err or 'empty'})", flush=True)
        return None
    newest = max(remote, key=_checkpoint_step)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ok, _, err = _gcloud_storage("cp", newest, str(checkpoint_dir))
    if not ok:
        print(f"WARNING: failed to fetch {newest} ({err})", flush=True)
        return None
    local = checkpoint_dir / Path(newest).name
    print(f"Fetched {newest} -> {local}", flush=True)
    return local


def is_torchrun() -> bool:
    """True when launched via torchrun / torch.distributed.launch."""
    return "LOCAL_RANK" in os.environ


def setup_ddp():
    """Initialize distributed process group and set CUDA device."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank


def cleanup_ddp():
    """Destroy the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model):
    """Unwrap torch.compile and DDP wrappers to access the raw module."""
    raw = model
    if hasattr(raw, "_orig_mod"):    # torch.compile
        raw = raw._orig_mod
    if hasattr(raw, "module"):       # DDP
        raw = raw.module
    return raw


def save_checkpoint(checkpoint_dir, step, model, optimizer, lr_scheduler,
                    save_lora=False, backend=None):
    """Save a training checkpoint to disk.

    Always saves the projector state dict, optimizer, and LR scheduler.
    When ``save_lora=True``, also saves LoRA adapter weights from the
    language model (used by instruct / multilingual pipelines).

    ``backend`` routes the write through the accelerator seam. Pass it and call
    this from EVERY rank: assembling the state dict reads tensors off the
    device, which on SPMD is collective work that every rank must join. The
    backend decides who writes. Omitting it keeps the original single-process
    behaviour, which is what the instruct/multilingual pipelines still use.
    """
    save_path = checkpoint_dir / f"checkpoint_{step}.pt"
    raw_model = _unwrap_model(model)
    state = {
        "step": step,
        "projector": raw_model.multi_modal_projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    if save_lora:
        state["lora_adapter"] = {
            k: v for k, v in raw_model.language_model.state_dict().items()
            if "lora_" in k
        }
    if backend is None:
        torch.save(state, save_path)
        wrote = True
    else:
        backend.save(state, save_path)
        wrote = backend.is_main

    # Only the rank that wrote the bytes mirrors them; see
    # `mirror_checkpoint_to_gcs` on why rank-gating is safe here specifically.
    if wrote:
        print(f"Saved checkpoint to {save_path}")
        mirror_checkpoint_to_gcs(save_path)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the path to the highest-step checkpoint in a directory, or None."""
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: _checkpoint_step(p.name))


def save_hf_model(model, processor: TinyAyaVisionProcessor, checkpoint_dir: Path, training_config=None) -> Path:
    """Merge LoRA into the base model and save in HuggingFace format.

    If ``training_config.merge_with_base_llm`` is True, additionally performs
    linear interpolation (LERP) of the VLM's LLM backbone with the
    original base LLM weights.

    Returns the output directory path.
    """
    print("Merging LoRA and saving HF-compatible model...")
    raw_model = _unwrap_model(model)
    raw_model.language_model = raw_model.language_model.merge_and_unload()

    # Optionally merge with original base LLM via weight interpolation
    if (
        training_config is not None
        and getattr(training_config, "merge_with_base_llm", False)
        and training_config.base_llm_name
    ):
        from scripts.merge_weights import build_merged_vlm_state, _load_original_llm

        alpha = training_config.merge_alpha
        print(f"Merging VLM backbone with '{training_config.base_llm_name}' (α={alpha})...")

        finetuned_state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
        original_llm_state = _load_original_llm(
            training_config.base_llm_name, device="cpu", dtype=torch.bfloat16,
        )
        merged_state = build_merged_vlm_state(original_llm_state, finetuned_state, alpha)
        raw_model.load_state_dict(merged_state, strict=False)
        del original_llm_state, finetuned_state, merged_state
        print("Weight merge complete.")

    hf_output_dir = checkpoint_dir / "hf_model"
    save_for_inference(raw_model, processor, hf_output_dir)
    print(f"Saved HF-compatible model to {hf_output_dir}")
    return hf_output_dir


def build_lr_scheduler(optimizer, training_config, full_dataset_len, per_gpu_batch_size, world_size):
    """Build a cosine LR scheduler with linear warmup.

    Computes total optimisation steps from the dataset size, batch
    configuration and number of epochs, then constructs a sequential
    scheduler: linear warmup followed by cosine decay.
    """
    full_loader_len = full_dataset_len // (per_gpu_batch_size * world_size)
    total_steps = training_config.num_epochs * full_loader_len // training_config.grad_acc_steps
    warmup_steps = int(total_steps * training_config.warmup_ratio)

    if training_config.lr_scheduler_type == "cosine":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8 / training_config.learning_rate, total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=training_config.learning_rate * 0.01,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
        )
    else:
        raise ValueError(f"Unsupported LR scheduler type: {training_config.lr_scheduler_type}")
