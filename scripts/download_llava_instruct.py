"""Download LLaVA-Instruct-150K / LLaVA-v1.5-mix665k dataset.

Downloads instruction-following JSONs from HuggingFace and (optionally) the
required images from their upstream sources.

Usage:
    # Original 150K only
    python scripts/download_llava_instruct.py --output-dir /data/llava-instruct

    # mix665k JSON + all images
    python scripts/download_llava_instruct.py --output-dir /data/llava-instruct \\
        --mix665k --download-images

    # Faster with 16 connections per file
    python scripts/download_llava_instruct.py --output-dir /data/llava-instruct \\
        --mix665k --download-images --num-connections 16

After running, the layout will be::

    <output-dir>/
        llava_instruct_150k.json
        llava_v1_5_mix665k.json      (with --mix665k)
        coco/train2017/               COCO train2017  (~19 GB)
        gqa/images/                   GQA images      (~20 GB)
        ocr_vqa/images/               OCR-VQA images  (~10 GB)
        textvqa/train_images/         TextVQA images  (~7 GB)
        vg/VG_100K/                   VisualGenome P1 (~10 GB)
        vg/VG_100K_2/                 VisualGenome P2 (~5 GB)
"""

import argparse
import os
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from huggingface_hub import hf_hub_download

_print_lock = Lock()


def _log(msg):
    """Thread-safe print."""
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Multi-connection download
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BACKOFF_BASE = 10  # seconds; actual wait = base * 2^attempt


def _retry_request(request_fn, max_retries=_MAX_RETRIES):
    """Call *request_fn* with exponential backoff on transient HTTP errors."""
    import requests

    for attempt in range(max_retries + 1):
        try:
            return request_fn()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status not in _RETRYABLE_STATUS or attempt == max_retries:
                raise
            wait = _BACKOFF_BASE * (2 ** attempt)
            _log(f"    HTTP {status}, retrying in {wait}s (attempt {attempt + 1}/{max_retries}) ...")
            time.sleep(wait)


def _download_range(url, start, end, part_path):
    """Download bytes [start, end] of *url* into *part_path*."""
    import requests

    def _do():
        headers = {"Range": f"bytes={start}-{end}"}
        resp = requests.get(url, headers=headers, stream=True, timeout=1800)
        resp.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    _retry_request(_do)


def _fast_download(url, dest, num_connections=8):
    """Download *url* to *dest* using parallel HTTP range requests.

    Falls back to a single-stream download when the server does not
    advertise ``Accept-Ranges`` or the file size is unknown.
    """
    import requests

    dest = Path(dest)
    if dest.exists():
        _log(f"  {dest.name} already exists, skipping download.")
        return

    # Probe for range-request support and total size (with retry)
    def _head():
        h = requests.head(url, timeout=60, allow_redirects=True)
        h.raise_for_status()
        return h
    head = _retry_request(_head)
    total = int(head.headers.get("Content-Length", 0))
    accepts_ranges = head.headers.get("Accept-Ranges", "none").lower() != "none"

    if total == 0 or not accepts_ranges or num_connections <= 1:
        _log(f"  Downloading {dest.name} (single stream) ...")
        tmp_dest = dest.parent / f".{dest.name}.tmp"
        try:
            def _get():
                resp = requests.get(url, stream=True, timeout=1800)
                resp.raise_for_status()
                with open(tmp_dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            _retry_request(_get)
            tmp_dest.rename(dest)
        except BaseException:
            tmp_dest.unlink(missing_ok=True)
            raise
        _log(f"  Saved {dest.name}")
        return

    _log(
        f"  Downloading {dest.name} "
        f"({total / 1e9:.1f} GB, {num_connections} connections) ..."
    )

    parts_dir = dest.parent / f".{dest.name}.parts"
    parts_dir.mkdir(exist_ok=True)
    tmp_dest = dest.parent / f".{dest.name}.tmp"

    chunk_size = total // num_connections
    ranges = []
    for i in range(num_connections):
        start = i * chunk_size
        end = total - 1 if i == num_connections - 1 else (i + 1) * chunk_size - 1
        ranges.append((i, start, end, parts_dir / f"part_{i:04d}"))

    try:
        with ThreadPoolExecutor(max_workers=num_connections) as pool:
            futs = {
                pool.submit(_download_range, url, s, e, str(p)): idx
                for idx, s, e, p in ranges
            }
            for f in as_completed(futs):
                f.result()  # propagate exceptions

        # Merge parts into a temp file, then atomically rename
        with open(tmp_dest, "wb") as out:
            for _, _, _, part_path in ranges:
                with open(part_path, "rb") as inp:
                    shutil.copyfileobj(inp, out)
        tmp_dest.rename(dest)
        _log(f"  Saved {dest.name}")
    finally:
        tmp_dest.unlink(missing_ok=True)
        shutil.rmtree(parts_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_members(zip_path, members, dest):
    """Extract a subset of members from a zip file (for ProcessPoolExecutor)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in members:
            zf.extract(name, dest)


def _extract_zip(zip_path, dest, cleanup=True):
    """Extract *zip_path* into *dest*, using system ``unzip`` when available."""
    zip_path = Path(zip_path)
    dest = Path(dest)

    if shutil.which("unzip"):
        _log(f"  Extracting {zip_path.name} with unzip ...")
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(dest)],
            check=True,
        )
    else:
        _log(f"  Extracting {zip_path.name} (parallel Python) ...")
        num_workers = os.cpu_count() or 1
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            dirs = {
                str(dest / os.path.dirname(n))
                for n in names
                if os.path.dirname(n)
            }
            for d in dirs:
                os.makedirs(d, exist_ok=True)
        chunks = [names[i::num_workers] for i in range(num_workers)]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = [
                pool.submit(_extract_members, str(zip_path), chunk, str(dest))
                for chunk in chunks
                if chunk
            ]
            for f in as_completed(futs):
                f.result()

    _log(f"  Extracted {zip_path.name}")

    if cleanup:
        zip_path.unlink()
        _log(f"  Deleted {zip_path.name}")


# ---------------------------------------------------------------------------
# Download + extract a single image source (runs in its own thread)
# ---------------------------------------------------------------------------

def _download_and_extract_source(url, zip_path, dest, num_connections=8, cleanup=True):
    """Download a zip with parallel connections, then extract it."""
    _fast_download(url, zip_path, num_connections=num_connections)
    _extract_zip(zip_path, dest, cleanup=cleanup)


def _download_ocr_vqa(output_dir, num_workers=64):
    """Download OCR-VQA images from Amazon URLs using dataset.json.

    The dataset.json is first fetched from Google Drive (via gdown) if missing,
    then all ~207K images are downloaded in parallel from Amazon CDN.
    """
    import gdown
    import requests

    ocr_base = Path(output_dir) / "ocr_vqa"
    images_dir = ocr_base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ensure dataset.json exists (download from Google Drive if needed)
    dataset_json = ocr_base / "images" / "dataset.json"
    if not dataset_json.exists():
        _log("Downloading OCR-VQA dataset.json from Google Drive ...")
        folder_url = "https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_"
        gdown.download_folder(folder_url, output=str(images_dir), quiet=True)

    if not dataset_json.exists():
        _log("✗ Could not obtain OCR-VQA dataset.json")
        return

    import json
    with open(dataset_json) as f:
        data = json.load(f)

    # Step 2: Figure out which images still need downloading
    to_download = []
    for key, entry in data.items():
        ext = os.path.splitext(entry["imageURL"])[1] or ".jpg"
        img_path = images_dir / f"{key}{ext}"
        if not img_path.exists():
            to_download.append((entry["imageURL"], str(img_path)))

    if not to_download:
        _log("ocr_vqa/images already complete, skipping.")
        return

    _log(f"Downloading {len(to_download)} OCR-VQA images ({num_workers} workers) ...")

    def _fetch_one(url_dest):
        url, dest_path = url_dest
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
        except Exception:
            return False

    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for ok in pool.map(_fetch_one, to_download):
            if ok:
                succeeded += 1
            else:
                failed += 1
            total = succeeded + failed
            if total % 10000 == 0:
                _log(f"    OCR-VQA progress: {total}/{len(to_download)} "
                     f"({failed} failed)")

    _log(f"  OCR-VQA done: {succeeded} downloaded, {failed} failed "
         f"(out of {len(to_download)})")


def _download_hf_json(repo_id, filename, output):
    """Download a single JSON file from a HuggingFace dataset repo."""
    json_path = output / filename
    if json_path.exists():
        _log(f"  {filename} already exists, skipping.")
        return json_path
    _log(f"Downloading {filename} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(output),
    )
    _log(f"  Saved to {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Image source definitions for mix665k
# ---------------------------------------------------------------------------
IMAGE_SOURCES = {
    "coco": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "check_dir": "coco/train2017",
        "extract_dest": "coco",      # zip contains train2017/ at root
        "zip_name": "train2017.zip",
    },
    "gqa": {
        "url": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
        "check_dir": "gqa/images",
        "extract_dest": "gqa",        # zip contains images/ at root
        "zip_name": "gqa_images.zip",
    },
    "textvqa": {
        "url": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
        "check_dir": "textvqa/train_images",
        "extract_dest": "textvqa",    # zip contains train_images/ at root
        "zip_name": "textvqa_images.zip",
    },
    "vg_1": {
        "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
        "check_dir": "vg/VG_100K",
        "extract_dest": "vg",
        "zip_name": "vg_images_1.zip",
    },
    "vg_2": {
        "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
        "check_dir": "vg/VG_100K_2",
        "extract_dest": "vg",
        "zip_name": "vg_images_2.zip",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--mix665k",
        action="store_true",
        help="Also download llava_v1_5_mix665k.json.",
    )
    parser.add_argument(
        "--download-coco",
        action="store_true",
        help="Download COCO train2017 images (~19 GB).",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download ALL image sources for mix665k "
             "(COCO, GQA, TextVQA, VisualGenome; ~70 GB total).",
    )
    parser.add_argument(
        "--num-connections",
        type=int,
        default=8,
        help="Number of parallel HTTP connections per file download (default: 8).",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # --- Determine which image sources to download ---------------------------
    if args.download_images:
        sources_to_download = list(IMAGE_SOURCES.keys())
    elif args.download_coco:
        sources_to_download = ["coco"]
    else:
        sources_to_download = []

    # --- Launch ALL downloads in parallel ------------------------------------
    # Each image source runs in its own thread, and within each thread the
    # download itself uses N parallel HTTP range-request connections.
    # JSONs are downloaded concurrently alongside image sources.
    num_source_threads = (
        len(sources_to_download) + 1 + (1 if args.mix665k else 0)
        + (1 if args.download_images else 0)  # OCR-VQA
    )
    with ThreadPoolExecutor(max_workers=max(num_source_threads, 1)) as pool:
        futures = {}

        # JSON downloads
        futures[pool.submit(
            _download_hf_json,
            "liuhaotian/LLaVA-Instruct-150K", "llava_instruct_150k.json", output,
        )] = "json:llava_instruct_150k.json"

        if args.mix665k:
            futures[pool.submit(
                _download_hf_json,
                "liuhaotian/LLaVA-Instruct-150K", "llava_v1_5_mix665k.json", output,
            )] = "json:llava_v1_5_mix665k.json"

        # Image downloads (each downloads with N connections + extracts)
        for key in sources_to_download:
            src = IMAGE_SOURCES[key]
            check_path = output / src["check_dir"]
            if check_path.exists() and any(check_path.iterdir()):
                _log(f"{src['check_dir']} already present, skipping.")
                continue
            zip_path = output / src["zip_name"]
            dest = output / src["extract_dest"]
            dest.mkdir(parents=True, exist_ok=True)
            futures[pool.submit(
                _download_and_extract_source,
                src["url"], zip_path, dest,
                num_connections=args.num_connections,
            )] = f"images:{key}"

        # OCR-VQA (Google Drive via gdown, runs in parallel with everything else)
        if args.download_images:
            futures[pool.submit(_download_ocr_vqa, output)] = "images:ocr_vqa"

        # Wait for all to finish
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                fut.result()
                _log(f"✓ Finished {label}")
            except Exception as exc:
                _log(f"✗ Failed {label}: {exc}")

    if not sources_to_download:
        _log(
            "\nNo images downloaded. Use --download-coco or --download-images.\n"
        )

    _log("Done.")


if __name__ == "__main__":
    main()
