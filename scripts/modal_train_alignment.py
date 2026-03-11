"""
Run alignment training on Modal with an A10 GPU.

Usage: modal run scripts/modal_train_alignment.py
"""

import modal

app = modal.App("tayavision-train-alignment")
volume = modal.Volume.from_name("tayavision-data")
models_volume = modal.Volume.from_name("tayavision-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "datasets",
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
    )
    .add_local_dir("config", remote_path="/root/project/config")
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("pipeline", remote_path="/root/project/pipeline")
    .add_local_dir("models", remote_path="/root/project/models")
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 24,
)
def train(resume_run_id: str | None = None):
    import sys
    sys.path.insert(0, "/root/project")

    from config.training_config import AlignmentConfig
    from config.model_config import TinyAyaVisionConfig
    from pipeline.train_alignment import main

    main(
        training_config=AlignmentConfig(),
        model_config=TinyAyaVisionConfig(),
        resume_run_id=resume_run_id,
    )


@app.local_entrypoint()
def run(resume_run_id: str = None):
    train.remote(resume_run_id=resume_run_id)
