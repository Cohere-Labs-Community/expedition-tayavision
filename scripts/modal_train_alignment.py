"""
Run alignment training on Modal.

Usage:
    modal run --detach scripts/modal_train_alignment.py
    modal run --detach scripts/modal_train_alignment.py --vision siglip
    MODAL_GPU=A100-80GB modal run --detach scripts/modal_train_alignment.py --vision moonvit
    modal run --detach scripts/modal_train_alignment.py --resume-run-id <id>
    modal run --detach scripts/modal_train_alignment.py --llm global --learning-rate 1e-3 --weight-decay 0.01
"""

import os

import modal

# Read GPU from MODAL_GPU env var so it can be set before the app is created.
# Usage: MODAL_GPU=A100-80GB modal run --detach scripts/modal_train_alignment.py --vision moonvit
GPU = os.environ.get("MODAL_GPU", "A10G")

app = modal.App("tayavision-train-alignment")
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
        "huggingface_hub",
        "tokenizers",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "numpy",
        "tqdm",
        "einops",
        "wandb",
        "hydra-core",
        "omegaconf",
        "pyyaml",
    )
    .add_local_dir("config", remote_path="/root/project/config")
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("pipeline", remote_path="/root/project/pipeline")
    .add_local_dir("models", remote_path="/root/project/models")
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 24,
)
def train(
    vision: str = "moonvit",
    llm: str = "base",
    resume_run_id: str | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
):
    import sys
    sys.path.insert(0, "/root/project")

    from hydra import compose, initialize_config_dir
    from pipeline.train_alignment import run

    overrides = [f"vision={vision}", f"llm={llm}"]
    if resume_run_id:
        overrides.append(f"resume={resume_run_id}")
    if learning_rate is not None:
        overrides.append(f"learning_rate={learning_rate}")
    if weight_decay is not None:
        overrides.append(f"weight_decay={weight_decay}")

    with initialize_config_dir(config_dir="/root/project/config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
        run(cfg)


@app.local_entrypoint()
def main(
    vision: str = "moonvit",
    llm: str = "base",
    resume_run_id: str = None,
    learning_rate: float = None,
    weight_decay: float = None,
):
    train.remote(
        vision=vision,
        llm=llm,
        resume_run_id=resume_run_id,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
