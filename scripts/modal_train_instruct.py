"""
Run instruction fine-tuning (Phase 2) on Modal.

Supports the Qwen3 pivot + script-conditioned token allocation pipeline:
- --backbone selects the LLM family (qwen3 or tiny_aya)
- --controller selects the script-conditioned controller variant (off, b1, b2, b3)
- --debug-steps caps the run at N steps with a small batch for fast smoke tests

Phase A (H1 diagnostic): controller="off"
Phase B (controller training): controller=b1 / b2 / b3

Usage:
    # Phase A instruct with Qwen3, controller off
    modal run --detach scripts/modal_train_instruct.py --backbone qwen3 --controller off

    # Phase B instruct with B2 (fertility + image complexity)
    modal run --detach scripts/modal_train_instruct.py --backbone qwen3 --controller b2

    # From a specific alignment checkpoint
    modal run --detach scripts/modal_train_instruct.py \\
        --backbone qwen3 --controller off \\
        --alignment-checkpoint /models/<run_id>/checkpoint_<step>.pt

    # Smoke test
    modal run scripts/modal_train_instruct.py --backbone qwen3 --controller off --debug-steps 50

    # Tiny Aya regression
    modal run --detach scripts/modal_train_instruct.py --backbone tiny_aya --controller off

    # Resume
    modal run --detach scripts/modal_train_instruct.py --backbone qwen3 --controller off --resume-run-id <id>
"""

import os

import modal

GPU = os.environ.get("MODAL_GPU", "A100-40GB")

app = modal.App("tayavision-train-instruct")
volume = modal.Volume.from_name("tayavision-data")
models_volume = modal.Volume.from_name("tayavision-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .uv_pip_install(
        "torch==2.9.1",
        "torchvision",
        "transformers==4.56.2",
        "accelerate",
        "datasets>=2.16.0",
        "huggingface_hub",
        "tokenizers",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "numpy",
        "tqdm",
        "einops",
        "wandb",
        "peft",
        "hydra-core",
        "omegaconf",
        "pyyaml",
    )
    .add_local_dir("config", remote_path="/root/project/config")
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("pipeline", remote_path="/root/project/pipeline")
    .add_local_dir("models", remote_path="/root/project/models")
)


def build_overrides(
    backbone: str,
    controller: str,
    training: str,
    alignment_checkpoint: str | None,
    resume_run_id: str | None,
    learning_rate: float | None,
    weight_decay: float | None,
    debug_steps: int | None,
) -> list[str]:
    """Build the Hydra override list for an instruct run."""
    overrides = [
        f"backbone={backbone}",
        f"controller={controller}",
        f"training={training}",
        # Use Modal volume paths for checkpoint saves
        "training.models_dir=/models",
    ]
    if alignment_checkpoint:
        overrides.append(f"training.alignment_checkpoint={alignment_checkpoint}")
    if resume_run_id:
        overrides.append(f"resume={resume_run_id}")
    if learning_rate is not None:
        overrides.append(f"training.learning_rate={learning_rate}")
    if weight_decay is not None:
        overrides.append(f"training.weight_decay={weight_decay}")
    if debug_steps is not None:
        # Smoke-test recipe from 2026-05-16_pivot_run_commands.md.
        # NOTE: `+debug.steps` requires the pipeline `train()` to read
        # cfg.debug.steps for early termination. See the run-commands doc.
        overrides.extend([
            "training.batch_size=4",
            "training.grad_acc_steps=1",
            "training.save_steps=20",
            f"+debug.steps={debug_steps}",
        ])
    return overrides


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 24,
)
def train(
    backbone: str = "qwen3",
    controller: str = "off",
    training: str = "instruct",
    alignment_checkpoint: str | None = None,
    resume_run_id: str | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    debug_steps: int | None = None,
):
    import sys
    sys.path.insert(0, "/root/project")

    from hydra import compose, initialize_config_dir
    from pipeline.train_instruct import run

    overrides = build_overrides(
        backbone=backbone,
        controller=controller,
        training=training,
        alignment_checkpoint=alignment_checkpoint,
        resume_run_id=resume_run_id,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        debug_steps=debug_steps,
    )

    print("=" * 60)
    print("Instruct training on Modal")
    print("=" * 60)
    print(f"GPU: {GPU}")
    print(f"Backbone: {backbone}")
    print(f"Controller: {controller}")
    print(f"Training preset: {training}")
    print(f"Alignment checkpoint: {alignment_checkpoint or '(from training yaml)'}")
    print(f"Debug steps: {debug_steps or 'off (full run)'}")
    print(f"Overrides: {overrides}")
    print("=" * 60)

    with initialize_config_dir(config_dir="/root/project/config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
        run(cfg)


@app.local_entrypoint()
def main(
    backbone: str = "qwen3",
    controller: str = "off",
    training: str = "instruct",
    alignment_checkpoint: str = None,
    resume_run_id: str = None,
    learning_rate: float = None,
    weight_decay: float = None,
    debug_steps: int = None,
):
    train.remote(
        backbone=backbone,
        controller=controller,
        training=training,
        alignment_checkpoint=alignment_checkpoint,
        resume_run_id=resume_run_id,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        debug_steps=debug_steps,
    )
