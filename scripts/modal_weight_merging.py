import os

import modal

GPU = os.environ.get("MODAL_GPU", "A10G")

app = modal.App("tayavision-weight-merging")
volume = modal.Volume.from_name("tayavision-data")
models_volume = modal.Volume.from_name("tayavision-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .uv_pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "tokenizers",
        "Pillow",
        "numpy",
        "tqdm",
        "wandb",
        "pyyaml",
        "safetensors"
    )
    .add_local_dir("config", remote_path="/root/project/config")
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("pipeline", remote_path="/root/project/pipeline")
    .add_local_dir("models", remote_path="/root/project/models")
    .add_local_dir("scripts", remote_path="/root/project/scripts")
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 24,
)
def merge_weights(
    original: str = "CohereLabs/tiny-aya-global",
    finetuned: str = "TrishanuDas/tayavision-instruct-665k",
    alpha: float = 0.5,
    output: str = "/models/merged",
    save_hf: bool = False,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
):
    import sys
    sys.path.insert(0, "/root/project")

    from scripts.merge_weights import main

    argv = [
        "--original", original,
        "--finetuned", finetuned,
        "--alpha", str(alpha),
        "--output", output,
        "--dtype", dtype,
        "--device", device,
    ]
    if save_hf:
        argv.append("--save-hf")

    main(argv=argv)

    if push_to_hub:
        from pathlib import Path
        from huggingface_hub import HfApi

        hf_dir = Path(output) / "hf_model"
        if not hf_dir.exists():
            raise RuntimeError(
                f"Cannot push to hub: {hf_dir} does not exist. "
                "Set save_hf=True to generate the HF model directory first."
            )

        repo_id = hub_repo_id or f"tayavision-merged-alpha-{alpha}"
        api = HfApi()
        repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=str(hf_dir),
            repo_id=repo_url.repo_id,
            commit_message=f"Upload merged weights (alpha={alpha})",
        )
        print(f"Pushed merged model to {repo_url}")


@app.local_entrypoint()
def main():
    merge_weights.remote(
        original="CohereLabs/tiny-aya-global",
        finetuned="TrishanuDas/tayavision-instruct-665k",
        alpha=0.5,
        output="/models/merged/tayavision_merged_alpha_0.5",
        save_hf=True,
        push_to_hub=True,
    )
