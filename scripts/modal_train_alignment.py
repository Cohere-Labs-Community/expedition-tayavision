"""
Run alignment training on Modal.

Usage:
    modal run --detach scripts/modal_train_alignment.py
    modal run --detach scripts/modal_train_alignment.py --vision siglip --gpu A10G
    modal run --detach scripts/modal_train_alignment.py --vision moonvit --gpu A100
    modal run --detach scripts/modal_train_alignment.py --resume-run-id <id>
"""

import modal

app = modal.App("tayavision-train-alignment")
volume = modal.Volume.from_name("tayavision-data")
models_volume = modal.Volume.from_name("tayavision-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
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


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/data": volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 24,
)
class Trainer:
    @modal.method()
    def train(self, vision: str = "moonvit", llm: str = "base", resume_run_id: str | None = None):
        import sys
        sys.path.insert(0, "/root/project")

        from hydra import compose, initialize_config_dir
        from pipeline.train_alignment import run

        overrides = [f"vision={vision}", f"llm={llm}"]
        if resume_run_id:
            overrides.append(f"resume={resume_run_id}")

        with initialize_config_dir(config_dir="/root/project/config", version_base="1.3"):
            cfg = compose(config_name="config", overrides=overrides)
            run(cfg)


@app.local_entrypoint()
def main(vision: str = "moonvit", llm: str = "base", resume_run_id: str = None, gpu: str = "A100"):
    Trainer.with_options(gpu=gpu)().train.remote(vision=vision, llm=llm, resume_run_id=resume_run_id)
