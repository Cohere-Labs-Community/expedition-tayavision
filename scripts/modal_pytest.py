"""
Run pytest on Modal with an A10 GPU.

Usage: modal run scripts/modal_pytest.py
"""

import modal
import subprocess
import sys

app = modal.App("tayavision-pytest")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "torchvision",
        "transformers==4.56.2",
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
        "pytest",
        "hydra-core",
        "omegaconf",
    )
    .add_local_dir("config", remote_path="/root/project/config")
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("models", remote_path="/root/project/models")
    .add_local_dir("pipeline", remote_path="/root/project/pipeline")
    .add_local_dir("tests", remote_path="/root/project/tests")
)


@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def run_tests():
    # Run pytest directly from the project root
    result = subprocess.run(
        ["pytest", "-v", "tests/test_vision_encoder.py"],
        cwd="/root/project",
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"Pytest failed with exit code {result.returncode}")
        sys.exit(1)


@app.local_entrypoint()
def run():
    run_tests.remote()
