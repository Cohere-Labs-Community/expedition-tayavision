"""
Download LLaVA-Pretrain to a Modal volume.

Usage: modal run scripts/modal_download.py
"""

import modal

app = modal.App("tayavision-download")
volume = modal.Volume.from_name("tayavision-data")

image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")

DATA_DIR = "/data/llava-pretrain"


@app.function(image=image, volumes={"/data": volume}, timeout=7200, ephemeral_disk=524_288)
def download():
    import json
    import os
    import zipfile
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    os.environ["HF_HUB_CACHE"] = "/data/.hf_cache"

    output = Path(DATA_DIR)
    images_dir = output / "images"

    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Data already exists at {DATA_DIR}, skipping.")
        return

    output.mkdir(parents=True, exist_ok=True)

    print("Downloading conversations JSON...")
    json_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="blip_laion_cc_sbu_558k.json",
        repo_type="dataset",
        local_dir=str(output),
    )
    with open(json_path) as f:
        convos = json.load(f)
    print(f"  {len(convos)} conversations")

    print("Downloading images.zip (~13GB, this will take a while)...")
    zip_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="images.zip",
        repo_type="dataset",
        local_dir=str(output),
    )

    print("Extracting images...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(output))
    print(f"  Extracted to {images_dir}")

    Path(zip_path).unlink()
    print("Deleted images.zip to save space.")

    volume.commit()
    print("Done.")


@app.local_entrypoint()
def main():
    download.remote()
