"""
Download LLaVA-Pretrain dataset to a local directory.

Downloads blip_laion_cc_sbu_558k.json (conversations) and images.zip,
then extracts images to disk.

Usage: python scripts/download_llava_pretrain.py --output-dir /data/llava-pretrain
"""

import argparse
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    images_dir = output / "images"

    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Data already exists at {output}, skipping.")
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
    print("Done.")


if __name__ == "__main__":
    main()
