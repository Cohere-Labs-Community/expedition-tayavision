"""Download LLaVA-Instruct-150K dataset.

Downloads the instruction-following JSON from HuggingFace. Images must be
sourced separately from COCO train2017 (~19 GB).

Usage:
    python scripts/download_llava_instruct.py --output-dir /data/llava-instruct

After running, symlink or copy the COCO images so the layout is::

    <output-dir>/
        llava_instruct_150k.json
        coco/
            train2017/
                000000000009.jpg
                ...
"""

import argparse
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


def _extract_members(zip_path, members, dest):
    """Extract a subset of members from a zip file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in members:
            zf.extract(name, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--download-coco",
        action="store_true",
        help="Also download COCO train2017 images (~19 GB).",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / "llava_instruct_150k.json"
    if json_path.exists():
        print(f"JSON already exists at {json_path}, skipping.")
    else:
        print("Downloading llava_instruct_150k.json ...")
        hf_hub_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            filename="llava_instruct_150k.json",
            repo_type="dataset",
            local_dir=str(output),
        )
        print(f"  Saved to {json_path}")

    coco_dir = output / "coco" / "train2017"
    if coco_dir.exists() and any(coco_dir.iterdir()):
        print(f"COCO images already present at {coco_dir}")
    elif args.download_coco:
        import requests

        coco_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output / "train2017.zip"

        if not zip_path.exists():
            url = "http://images.cocodataset.org/zips/train2017.zip"
            print(f"Downloading COCO train2017 from {url} ...")
            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Downloaded to {zip_path}")

        print("Extracting images (multi-core)...")
        dest = output / "coco"
        num_workers = os.cpu_count() or 1
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            # Pre-create all directories to avoid race conditions between workers
            dirs = {str(dest / os.path.dirname(n)) for n in names if os.path.dirname(n)}
            for d in dirs:
                os.makedirs(d, exist_ok=True)
        # Split file list into chunks, one per worker
        chunks = [names[i::num_workers] for i in range(num_workers)]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(_extract_members, str(zip_path), chunk, str(dest))
                for chunk in chunks
                if chunk
            ]
            for fut in as_completed(futures):
                fut.result()
        print(f"  Extracted to {coco_dir}")

        zip_path.unlink()
        print("Deleted train2017.zip to save space.")
    else:
        print(
            f"\nCOCO train2017 images are required but not found at {coco_dir}.\n"
            "Either:\n"
            "  1. Re-run with --download-coco  (downloads ~19 GB)\n"
            f"  2. Symlink existing images:  ln -s /path/to/coco {output / 'coco'}\n"
        )

    print("Done.")


if __name__ == "__main__":
    main()
