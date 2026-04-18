"""
Download LLaVA datasets to a Modal volume.

LLaVA-Pretrain: blip_laion_cc_sbu_558k.json + images.zip (~27GB from HuggingFace).
LLaVA-Instruct-150K: llava_instruct_150k.json (HuggingFace) + COCO train2017
images (~18GB from images.cocodataset.org). Both zips are stored unextracted to
stay under the Modal Volume 500K inode limit; extraction happens at training time.

Usage:
    modal run scripts/modal_download.py                        # pretrain (default)
    modal run scripts/modal_download.py --dataset instruct
    modal run scripts/modal_download.py --dataset all
"""

import modal

app = modal.App("tayavision-download")
volume = modal.Volume.from_name("tayavision-data")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub", "hf_transfer"
)

PRETRAIN_DIR = "/data/llava-pretrain"
INSTRUCT_DIR = "/data/llava-instruct"


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,
    ephemeral_disk=524_288,
    secrets=[modal.Secret.from_name("huggingface")],
)
def download_pretrain():
    import json
    import os
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    output = Path(PRETRAIN_DIR)
    zip_dest = output / "images.zip"
    json_dest = output / "blip_laion_cc_sbu_558k.json"

    if zip_dest.exists() and json_dest.exists():
        print(f"Pretrain data already exists at {PRETRAIN_DIR}, skipping.")
        return

    output.mkdir(parents=True, exist_ok=True)

    if not json_dest.exists():
        print("Downloading blip_laion_cc_sbu_558k.json...")
        hf_hub_download(
            repo_id="liuhaotian/LLaVA-Pretrain",
            filename="blip_laion_cc_sbu_558k.json",
            repo_type="dataset",
            local_dir=str(output),
        )
        with open(json_dest) as f:
            convos = json.load(f)
        print(f"  {len(convos)} conversations")
    else:
        print("JSON already present, skipping.")

    if not zip_dest.exists():
        print("Downloading images.zip (~27GB, this will take a while)...")
        hf_hub_download(
            repo_id="liuhaotian/LLaVA-Pretrain",
            filename="images.zip",
            repo_type="dataset",
            local_dir=str(output),
        )
        print(f"  Saved to {zip_dest}")
    else:
        print("images.zip already present, skipping.")

    volume.commit()
    print("Pretrain download complete.")


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,
    ephemeral_disk=524_288,
    secrets=[modal.Secret.from_name("huggingface")],
)
def download_instruct():
    import os
    import urllib.request
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    output = Path(INSTRUCT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    json_dest = output / "llava_instruct_150k.json"
    if not json_dest.exists():
        print("Downloading llava_instruct_150k.json...")
        hf_hub_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            filename="llava_instruct_150k.json",
            repo_type="dataset",
            local_dir=str(output),
        )
        print(f"  Saved to {json_dest}")
    else:
        print("llava_instruct_150k.json already present, skipping.")

    # LLaVA-Instruct-150K has no images; they come from COCO train2017.
    coco_zip = output / "coco_train2017.zip"
    if not coco_zip.exists():
        print("Downloading COCO train2017 images (~18GB, this will take a while)...")
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/zips/train2017.zip",
            str(coco_zip),
        )
        print(f"  Saved to {coco_zip}")
    else:
        print("coco_train2017.zip already present, skipping.")

    volume.commit()
    print("Instruct download complete.")


@app.local_entrypoint()
def main(dataset: str = "pretrain"):
    if dataset not in ("pretrain", "instruct", "all"):
        raise ValueError(f"--dataset must be one of: pretrain, instruct, all. Got: {dataset!r}")
    if dataset in ("pretrain", "all"):
        download_pretrain.remote()
    if dataset in ("instruct", "all"):
        download_instruct.remote()
