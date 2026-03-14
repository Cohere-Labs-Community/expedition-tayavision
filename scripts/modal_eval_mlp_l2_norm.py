import modal

app = modal.App("tayavision-eval-mlp-l2-norm")

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
def evaluate(num_per_class: int = 1):
    import sys
    sys.path.insert(0, "/root/project")
    from evaluation.eval_mlp_l2_norm import main
    main(num_per_class=num_per_class)


@app.local_entrypoint()
def run(num_per_class: int = 1):
    evaluate.remote(num_per_class=num_per_class)
