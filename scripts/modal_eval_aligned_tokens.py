import modal

app = modal.App("tayavision-eval-alignment")

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
    )
    .add_local_dir("config", remote_path="/root/project/config")
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("evaluation", remote_path="/root/project/evaluation")
    .add_local_dir("models", remote_path="/root/project/models")
)


@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def evaluate(top_k: int = 10):
    import sys
    sys.path.insert(0, "/root/project")
    from evaluation.eval_aligned_tokens import main
    return main(top_k=top_k)


@app.local_entrypoint()
def run(top_k: int = 10):
    from evaluation.eval_aligned_tokens import save_assets

    results = evaluate.remote(top_k=top_k)
    save_assets(results)
