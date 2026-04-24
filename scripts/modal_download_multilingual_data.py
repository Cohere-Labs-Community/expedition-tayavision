"""
Download multilingual datasets to a Modal volume.

Downloads these datasets to the `multilingual-data` Modal volume:
- neulab/PangeaInstruct (JSON + optional image archives)
- MBZUAI/palo_multilingual_dataset (JSON)
- liuhaotian/LLaVA-Instruct-150K (JSON)
- CohereForAI/aya_dataset (HF Dataset)
- MBZUAI/ALM-Bench (HF Dataset with embedded images)
- WueNLP/mvl-sib (TSV + topic images)
- sil-ai/bloom-lm (JSON, gated - requires HF token)
- wikimedia/wit_base (Parquet shards with embedded images)
- sil-ai/bloom-captioning (HF Dataset, gated - requires HF token)

Prerequisites:
    Create a Modal secret named "huggingface" with your HF_TOKEN:
        modal secret create huggingface HF_TOKEN=hf_xxx...

Usage:
    # Download all datasets (excluding Pangea images and bloom-captioning)
    modal run scripts/modal_download_multilingual_data.py

    # Download specific dataset(s)
    modal run scripts/modal_download_multilingual_data.py --datasets "pangea,aya,alm-bench"

    # Download with Pangea image archives
    modal run scripts/modal_download_multilingual_data.py --download-images

    # Download bloom-captioning (gated dataset)
    modal run scripts/modal_download_multilingual_data.py --datasets bloom-captioning

    # Configure WIT shards (default: 10 shards ≈ 2M samples)
    modal run scripts/modal_download_multilingual_data.py --datasets wit --wit-shards 50

Volume structure after download:
    /data/
        pangea-instruct/
            PangeaIns.json
            images/  (if --download-images)
        palo/
            palo_multilingual_dataset.json
        llava-instruct/
            llava_instruct_150k.json
        aya-dataset/
            (HF Dataset format)
        alm-bench/
            (HF Dataset format)
        mvl-sib/
            tsv/
            images/
        bloom-lm/
            bloom_lm_train.json
        wit/
            (Parquet shards)
        bloom-captioning/
            (HF Dataset format)

Note: COCO images (used by PALO and LLaVA-Instruct) are NOT downloaded by this
script. Use scripts/modal_download.py for LLaVA-Pretrain which includes COCO.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Literal

import modal

# ---------------------------------------------------------------------------
# Modal app configuration
# ---------------------------------------------------------------------------

app = modal.App("multilingual-data-download")
volume = modal.Volume.from_name("multilingual-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.20.0",
        "datasets>=2.16.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "orjson>=3.9.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
    )
)

DATA_DIR = Path("/data")

# Thread-safe logging
_print_lock = Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "pangea": {
        "name": "PangeaInstruct",
        "hf_repo": "neulab/PangeaInstruct",
        "output_dir": "pangea-instruct",
        "type": "json",
        "files": ["PangeaIns.json"],
    },
    "palo": {
        "name": "PALO Multilingual",
        "hf_repo": "MBZUAI/palo_multilingual_dataset",
        "output_dir": "palo",
        "type": "json",
        "files": ["palo_multilingual_dataset.json"],
    },
    "llava-instruct": {
        "name": "LLaVA-Instruct-150K",
        "hf_repo": "liuhaotian/LLaVA-Instruct-150K",
        "output_dir": "llava-instruct",
        "type": "json",
        "files": ["llava_instruct_150k.json"],
    },
    "aya": {
        "name": "Aya Dataset",
        "hf_repo": "CohereForAI/aya_dataset",
        "output_dir": "aya-dataset",
        "type": "hf_dataset",
        "split": "train",
    },
    "alm-bench": {
        "name": "ALM-Bench",
        "hf_repo": "MBZUAI/ALM-Bench",
        "output_dir": "alm-bench",
        "type": "hf_dataset",
        "split": "test",
    },
    "mvl-sib": {
        "name": "MVL-SIB",
        "hf_repo": "WueNLP/mvl-sib",
        "output_dir": "mvl-sib",
        "type": "custom",
    },
    "bloom-lm": {
        "name": "Bloom LM",
        "hf_repo": "sil-ai/bloom-lm",
        "output_dir": "bloom-lm",
        "type": "json",
        "files": ["bloom_lm_train.json"],
        "gated": True,
    },
    "wit": {
        "name": "WIT Base",
        "hf_repo": "wikimedia/wit_base",
        "output_dir": "wit",
        "type": "parquet_shards",
        "total_shards": 330,
    },
    "bloom-captioning": {
        "name": "Bloom Captioning",
        "hf_repo": "sil-ai/bloom-captioning",
        "output_dir": "bloom-captioning",
        "type": "hf_dataset",
        "split": "test",
        "gated": True,
    },
}

# Default datasets to download (excludes gated bloom-captioning)
DEFAULT_DATASETS = [
    "pangea", "palo", "llava-instruct", "aya", "alm-bench",
    "mvl-sib", "bloom-lm", "wit",
]

# Pangea image subsets (downloaded only with --download-images)
PANGEA_IMAGE_SUBSETS = [
    "general/COCO",
    "general/MSCOCO",
    "general/VG",
    "general/gqa",
    "general/MTVQA",
    "general/ShareGPT4V",
    "cultural/laion-cultural-150k",
    "caption/STAIR-Captions",
    "doc+chart/ChartQA",
]


# ---------------------------------------------------------------------------
# Multi-connection HTTP download (adapted from download_multilingual.py)
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BACKOFF_BASE = 10


def _retry_request(request_fn, max_retries: int = _MAX_RETRIES):
    """Call request_fn with exponential backoff on transient HTTP errors."""
    import time
    import requests

    for attempt in range(max_retries + 1):
        try:
            return request_fn()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status not in _RETRYABLE_STATUS or attempt == max_retries:
                raise
            wait = _BACKOFF_BASE * (2 ** attempt)
            _log(f"    HTTP {status}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt == max_retries:
                raise
            wait = _BACKOFF_BASE * (2 ** attempt)
            _log(f"    Connection error, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)


def _download_range(url: str, start: int, end: int, part_path: Path) -> None:
    """Download bytes [start, end] of url into part_path."""
    import requests

    def _do():
        headers = {"Range": f"bytes={start}-{end}"}
        resp = requests.get(url, headers=headers, stream=True, timeout=1800)
        resp.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    _retry_request(_do)


def _fast_download(url: str, dest: Path, num_connections: int = 8) -> None:
    """Download url to dest using parallel HTTP range requests."""
    import requests

    if dest.exists():
        _log(f"  {dest.name} already exists, skipping download.")
        return

    # Probe for range-request support
    def _head():
        h = requests.head(url, timeout=60, allow_redirects=True)
        h.raise_for_status()
        return h

    head = _retry_request(_head)
    total = int(head.headers.get("Content-Length", 0))
    accepts_ranges = head.headers.get("Accept-Ranges", "none").lower() != "none"

    if total == 0 or not accepts_ranges or num_connections <= 1:
        _log(f"  Downloading {dest.name} (single stream)...")
        tmp = dest.parent / f".{dest.name}.tmp"
        try:
            def _get():
                resp = requests.get(url, stream=True, timeout=1800)
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            _retry_request(_get)
            tmp.rename(dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        _log(f"  Saved {dest.name}")
        return

    _log(f"  Downloading {dest.name} ({total / 1e9:.1f} GB, {num_connections} connections)...")

    parts_dir = dest.parent / f".{dest.name}.parts"
    parts_dir.mkdir(exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp"

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
                f.result()

        # Merge parts atomically
        with open(tmp, "wb") as out:
            for _, _, _, part_path in ranges:
                with open(part_path, "rb") as inp:
                    shutil.copyfileobj(inp, out, length=1 << 20)
        tmp.rename(dest)
        _log(f"  Saved {dest.name}")
    finally:
        tmp.unlink(missing_ok=True)
        shutil.rmtree(parts_dir, ignore_errors=True)


def _hf_resolve_url(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Resolve the direct download URL for a file in a HF repo."""
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def _extract_tar(archive_path: Path, dest_dir: Path, cleanup: bool = True) -> None:
    """Extract tar/tar.gz using system tar (faster) or Python tarfile."""
    if shutil.which("tar"):
        _log(f"  Extracting {archive_path.name} with tar...")
        subprocess.run(
            ["tar", "-xf", str(archive_path), "-C", str(dest_dir)],
            check=True,
        )
    else:
        _log(f"  Extracting {archive_path.name} with Python...")
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(str(dest_dir), filter="data")
    _log(f"  Extracted {archive_path.name} → {dest_dir}")
    if cleanup:
        archive_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dataset download functions
# ---------------------------------------------------------------------------

def download_json_dataset(
    config: dict,
    output_base: Path,
    num_connections: int = 8,
    hf_token: str | None = None,
) -> bool:
    """Download JSON file(s) from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    output_dir = output_base / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[{config['name']}] Downloading to {output_dir}...")

    for filename in config["files"]:
        dest = output_dir / filename
        if dest.exists():
            _log(f"  {filename} already exists, skipping.")
            continue

        # Use hf_hub_download for authentication support (gated datasets)
        try:
            downloaded_path = hf_hub_download(
                config["hf_repo"],
                filename,
                repo_type="dataset",
                cache_dir=str(output_base / ".hf_cache"),
                local_dir=str(output_dir),
                token=hf_token,
            )
            _log(f"  Downloaded {filename}")
        except Exception as e:
            _log(f"  Failed to download {filename}: {e}")
            # Fallback to direct URL download (for public datasets)
            url = _hf_resolve_url(config["hf_repo"], filename)
            _fast_download(url, dest, num_connections=num_connections)

    _log(f"[{config['name']}] Done.")
    return True


def download_hf_dataset(
    config: dict,
    output_base: Path,
    hf_token: str | None = None,
) -> bool:
    """Download HuggingFace Dataset using datasets library."""
    from datasets import load_dataset

    output_dir = output_base / config["output_dir"]

    # Check if already downloaded
    if (output_dir / "dataset_info.json").exists():
        _log(f"[{config['name']}] Already downloaded at {output_dir}, skipping.")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[{config['name']}] Downloading from HuggingFace Hub...")

    try:
        ds = load_dataset(
            config["hf_repo"],
            split=config.get("split", "train"),
            cache_dir=str(output_base / ".hf_cache"),
            trust_remote_code=True,
            token=hf_token,
        )
        _log(f"[{config['name']}] Loaded {len(ds):,} samples, saving to disk...")
        ds.save_to_disk(str(output_dir))
        _log(f"[{config['name']}] Saved to {output_dir}")
        return True
    except Exception as e:
        _log(f"[{config['name']}] Failed: {e}")
        if config.get("gated"):
            _log(f"  This is a gated dataset. Provide --hf-secret with your HF token.")
        return False


def download_mvl_sib(output_base: Path, hf_token: str | None = None) -> bool:
    """Download MVL-SIB TSV files and topic images."""
    from huggingface_hub import HfApi, hf_hub_download

    config = DATASET_CONFIGS["mvl-sib"]
    output_dir = output_base / config["output_dir"]
    tsv_dir = output_dir / "tsv"
    images_dir = output_dir / "images"

    # Check if already downloaded
    if images_dir.exists() and any(images_dir.iterdir()):
        _log(f"[MVL-SIB] Already downloaded at {output_dir}, skipping.")
        return True

    tsv_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    _log("[MVL-SIB] Downloading...")
    cache_dir = str(output_base / ".hf_cache")

    # Download topic images (7 categories x 10 images = 70 total)
    _log("  Downloading topic images...")
    categories = [
        "entertainment", "geography", "health", "politics",
        "science_technology", "sports", "travel",
    ]
    fetched, failed = 0, 0
    for cat in categories:
        for idx in range(10):
            img_name = f"{cat}_{idx}.jpg"
            try:
                img_path = hf_hub_download(
                    config["hf_repo"],
                    f"data/images/sib200/{img_name}",
                    repo_type="dataset",
                    cache_dir=cache_dir,
                    local_dir=str(images_dir),
                    token=hf_token,
                )
                fetched += 1
            except Exception:
                failed += 1
    _log(f"  Downloaded {fetched} images, {failed} missing")

    # Download TSV files for each language
    _log("  Downloading TSV files...")
    api = HfApi()
    try:
        all_dirs = [
            f.path.split("/")[-1]
            for f in api.list_repo_tree(
                config["hf_repo"],
                repo_type="dataset",
                path_in_repo="data/sib200",
            )
        ]
    except Exception as e:
        _log(f"  Failed to list languages: {e}")
        return False

    downloaded = 0
    for lang_dir in all_dirs:
        try:
            lang_tsv_dir = tsv_dir / lang_dir
            lang_tsv_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                config["hf_repo"],
                f"data/sib200/{lang_dir}/train.tsv",
                repo_type="dataset",
                cache_dir=cache_dir,
                local_dir=str(lang_tsv_dir),
                token=hf_token,
            )
            downloaded += 1
        except Exception:
            continue

    _log(f"[MVL-SIB] Downloaded {downloaded} language TSV files.")
    return True


def download_wit(
    output_base: Path,
    num_shards: int = 10,
    num_connections: int = 8,
    hf_token: str | None = None,
) -> bool:
    """Download WIT parquet shards."""
    from huggingface_hub import hf_hub_download

    config = DATASET_CONFIGS["wit"]
    output_dir = output_base / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[WIT] Downloading {num_shards} parquet shards to {output_dir}...")

    downloaded = 0
    for shard_idx in range(num_shards):
        shard_name = f"train-{shard_idx:05d}-of-00330.parquet"
        dest = output_dir / shard_name

        if dest.exists():
            _log(f"  Shard {shard_idx} already exists, skipping.")
            downloaded += 1
            continue

        try:
            hf_hub_download(
                config["hf_repo"],
                f"data/{shard_name}",
                repo_type="dataset",
                cache_dir=str(output_base / ".hf_cache"),
                local_dir=str(output_dir),
                token=hf_token,
            )
            downloaded += 1
            _log(f"  Downloaded shard {shard_idx}")
        except Exception as e:
            _log(f"  Failed to download shard {shard_idx}: {e}")

    _log(f"[WIT] Downloaded {downloaded}/{num_shards} shards.")
    return downloaded > 0


def download_pangea_images(
    output_base: Path,
    subsets: list[str] | None = None,
    num_connections: int = 8,
    hf_token: str | None = None,
) -> bool:
    """Download and extract Pangea image archives."""
    from huggingface_hub import HfApi, hf_hub_download

    config = DATASET_CONFIGS["pangea"]
    output_dir = output_base / config["output_dir"]
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    subsets = subsets or PANGEA_IMAGE_SUBSETS
    _log(f"[Pangea Images] Downloading {len(subsets)} image subsets...")

    api = HfApi(token=hf_token)
    try:
        repo_files = [
            f.path for f in api.list_repo_tree(
                config["hf_repo"],
                repo_type="dataset",
                recursive=True,
            )
        ]
    except Exception as e:
        _log(f"  Failed to list repo files: {e}")
        return False

    for subset in subsets:
        # Find archive files for this subset
        archive_files = [
            f for f in repo_files
            if f.startswith(subset + "/") and (
                f.endswith(".tar") or f.endswith(".zip") or f.endswith(".tar.gz")
            )
        ]
        # Skip multi-part files
        archive_files = [
            f for f in archive_files
            if not f.endswith((".partaa", ".partab"))
        ]

        if not archive_files:
            _log(f"  [{subset}] No archives found, skipping")
            continue

        subset_dir = images_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Check if already extracted
        existing = [
            f for f in subset_dir.iterdir()
            if not f.name.endswith((".tar", ".zip", ".tar.gz", ".tmp"))
        ] if subset_dir.exists() else []
        if existing:
            _log(f"  [{subset}] Already extracted ({len(existing)} items), skipping")
            continue

        for fname in archive_files:
            local_path = images_dir / fname
            local_path.parent.mkdir(parents=True, exist_ok=True)

            _log(f"  [{subset}] Downloading {Path(fname).name}...")
            try:
                hf_hub_download(
                    config["hf_repo"],
                    fname,
                    repo_type="dataset",
                    cache_dir=str(output_base / ".hf_cache"),
                    local_dir=str(images_dir),
                    token=hf_token,
                )
            except Exception as e:
                _log(f"  [{subset}] Failed to download {fname}: {e}")
                continue
            _extract_tar(local_path, subset_dir, cleanup=True)

        _log(f"  [{subset}] Done")

    _log("[Pangea Images] Finished.")
    return True


# ---------------------------------------------------------------------------
# Main download orchestrator
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=14400,  # 4 hours
    ephemeral_disk=524_288,  # 512GB for large datasets
    secrets=[modal.Secret.from_name("huggingface")],
)
def download_datasets(
    datasets: list[str],
    download_images: bool = False,
    wit_shards: int = 10,
    num_connections: int = 8,
) -> dict[str, bool]:
    """Download specified datasets to the Modal volume.

    Args:
        datasets: List of dataset keys to download (from DATASET_CONFIGS)
        download_images: Whether to download Pangea image archives
        wit_shards: Number of WIT parquet shards to download (default: 10)
        num_connections: Number of parallel connections for large downloads

    Returns:
        Dict mapping dataset name to success status
    """
    os.environ["HF_HUB_CACHE"] = str(DATA_DIR / ".hf_cache")
    (DATA_DIR / ".hf_cache").mkdir(parents=True, exist_ok=True)

    # Get HF token from Modal secret (injected as environment variable)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        _log("HuggingFace token found in environment")
    else:
        _log("Warning: No HF_TOKEN found. Gated datasets may fail.")

    results: dict[str, bool] = {}

    for ds_key in datasets:
        if ds_key not in DATASET_CONFIGS:
            _log(f"Unknown dataset: {ds_key}")
            results[ds_key] = False
            continue

        config = DATASET_CONFIGS[ds_key]

        try:
            if config["type"] == "json":
                results[ds_key] = download_json_dataset(
                    config, DATA_DIR, num_connections=num_connections, hf_token=hf_token
                )
            elif config["type"] == "hf_dataset":
                results[ds_key] = download_hf_dataset(
                    config, DATA_DIR, hf_token=hf_token
                )
            elif ds_key == "mvl-sib":
                results[ds_key] = download_mvl_sib(DATA_DIR, hf_token=hf_token)
            elif ds_key == "wit":
                results[ds_key] = download_wit(
                    DATA_DIR, num_shards=wit_shards, num_connections=num_connections,
                    hf_token=hf_token
                )
            else:
                _log(f"Unknown download type for {ds_key}")
                results[ds_key] = False
        except Exception as e:
            _log(f"[{config['name']}] Error: {e}")
            results[ds_key] = False

    # Download Pangea images if requested
    if download_images and "pangea" in datasets:
        try:
            results["pangea-images"] = download_pangea_images(
                DATA_DIR, num_connections=num_connections, hf_token=hf_token
            )
        except Exception as e:
            _log(f"[Pangea Images] Error: {e}")
            results["pangea-images"] = False

    # Commit volume
    _log("\nCommitting volume...")
    volume.commit()

    # Summary
    _log("\n" + "=" * 60)
    _log("Download Summary:")
    _log("=" * 60)
    for ds_name, success in results.items():
        status = "✓" if success else "✗"
        _log(f"  {status} {ds_name}")

    return results


@app.local_entrypoint()
def main(
    datasets: str = "",
    download_images: bool = False,
    wit_shards: int = 10,
    num_connections: int = 8,
):
    """Download multilingual datasets to Modal volume.

    The HuggingFace token is automatically loaded from the Modal secret
    named "huggingface" (expects HF_TOKEN environment variable).

    Args:
        datasets: Comma-separated list of datasets to download. If empty,
                  downloads all default datasets (excludes bloom-captioning).
                  Available: pangea, palo, llava-instruct, aya, alm-bench,
                             mvl-sib, bloom-lm, wit, bloom-captioning
        download_images: Download Pangea image archives (large, ~100GB+)
        wit_shards: Number of WIT parquet shards to download (max 330)
        num_connections: Number of parallel HTTP connections for downloads
    """
    # Parse dataset list
    if datasets:
        dataset_list = [d.strip() for d in datasets.split(",")]
    else:
        dataset_list = DEFAULT_DATASETS.copy()

    print(f"Datasets to download: {dataset_list}")
    print(f"Download Pangea images: {download_images}")
    print(f"WIT shards: {wit_shards}")
    print("HuggingFace token: loaded from Modal secret 'huggingface'")

    # Run download
    results = download_datasets.remote(
        datasets=dataset_list,
        download_images=download_images,
        wit_shards=wit_shards,
        num_connections=num_connections,
    )

    # Print local summary
    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    success_count = sum(1 for v in results.values() if v)
    print(f"  {success_count}/{len(results)} datasets downloaded successfully")

    if not all(results.values()):
        print("\nFailed datasets:")
        for ds_name, success in results.items():
            if not success:
                print(f"  - {ds_name}")
