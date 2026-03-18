"""
Run instruction fine-tuning on Modal.

Usage:
    modal run --detach scripts/modal_train_instruct.py
    modal run --detach scripts/modal_train_instruct.py --alignment-checkpoint /models/<run_id>/checkpoint_<step>.pt
    modal run --detach scripts/modal_train_instruct.py --resume-run-id <id>
    MODAL_GPU=A100-80GB modal run --detach scripts/modal_train_instruct.py
    modal run --detach scripts/modal_train_instruct.py --learning-rate 1e-5 --weight-decay 0.01
"""

import os

import modal

GPU = os.environ.get("MODAL_GPU", "A10G")

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
    resume_run_id: str | None = None,
    alignment_checkpoint: str | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
):
    import sys
    sys.path.insert(0, "/root/project")

    from config.lora_config import LoraAdapterConfig
    from config.model_config import TinyAyaVisionConfig
    from config.training_config import InstructConfig
    from pipeline.train_instruct import main

    model_config = TinyAyaVisionConfig.for_global()
    lora_config = LoraAdapterConfig.from_vlm_config(model_config)

    training_config = InstructConfig()
    if alignment_checkpoint is not None:
        training_config.alignment_checkpoint = alignment_checkpoint
    if learning_rate is not None:
        training_config.learning_rate = learning_rate
    if weight_decay is not None:
        training_config.weight_decay = weight_decay

    main(
        training_config=training_config,
        model_config=model_config,
        lora_config=lora_config,
        resume_run_id=resume_run_id,
    )


@app.local_entrypoint()
def main(
    resume_run_id: str = None,
    alignment_checkpoint: str = None,
    learning_rate: float = None,
    weight_decay: float = None,
):
    train.remote(
        resume_run_id=resume_run_id,
        alignment_checkpoint=alignment_checkpoint,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
