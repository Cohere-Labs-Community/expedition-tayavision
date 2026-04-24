"""Download multilingual datasets for Phase 2 instruction tuning.

Downloads:
  1. PangeaIns — instructions JSON + image tarballs from neulab/PangeaInstruct
  2. Aya Dataset — pre-downloads parquet from HuggingFace Hub

All file downloads use parallel HTTP range requests (configurable connections
per file). Multiple independent sources download concurrently in separate
threads. Archive extraction uses system tar/unzip when available, falling
back to parallel Python extraction.

Usage:
    # Download Pangea JSON + Aya (fastest, text-only training)
    python scripts/download_multilingual.py --output-dir /data/multilingual

    # Download everything including images (16 connections per file)
    python scripts/download_multilingual.py --output-dir /data/multilingual \\
        --download-images --num-connections 16

    # Download specific image subsets only
    python scripts/download_multilingual.py --output-dir /data/multilingual \\
        --download-images --image-subsets general/COCO general/MTVQA
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from huggingface_hub import hf_hub_download, HfApi


PANGEA_REPO = "neulab/PangeaInstruct"

# Core image subsets to download (covers most instruction data).
DEFAULT_IMAGE_SUBSETS = [
    "general/COCO",                  # COCO train2017 (~19 GB)
    "general/MSCOCO",                # MSCOCO
    "general/VG",                    # VisualGenome
    "general/gqa",                   # GQA
    "general/MTVQA",                 # Multilingual text-centric VQA
    "general/ShareGPT4V",            # ShareGPT4V
    "cultural/laion-cultural-150k",  # Culturally diverse LAION
    "caption/STAIR-Captions",        # Japanese captions
    "doc+chart/ChartQA",             # Chart QA
]

# ---------------------------------------------------------------------------
# Thread-safe logging
# ---------------------------------------------------------------------------

_print_lock = Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Multi-connection HTTP range download (from download_llava_instruct.py)
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BACKOFF_BASE = 10


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
            _log(f"    HTTP {status}, retrying in {wait}s "
                 f"(attempt {attempt + 1}/{max_retries}) ...")
            time.sleep(wait)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            if attempt == max_retries:
                raise
            wait = _BACKOFF_BASE * (2 ** attempt)
            _log(f"    Connection error, retrying in {wait}s "
                 f"(attempt {attempt + 1}/{max_retries}) ...")
            time.sleep(wait)


def _download_range(url, start, end, part_path):
    """Download bytes [start, end] of *url* into *part_path*."""
    import requests

    def _do():
        headers = {"Range": f"bytes={start}-{end}"}
        resp = requests.get(url, headers=headers, stream=True, timeout=1800)
        resp.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                f.write(chunk)

    _retry_request(_do)


def _fast_download(url, dest, num_connections=8):
    """Download *url* to *dest* using parallel HTTP range requests.

    Falls back to single-stream if the server doesn't support ranges.
    Uses atomic rename via a .tmp file to prevent partial downloads.
    """
    import requests

    dest = Path(dest)
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
        _log(f"  Downloading {dest.name} (single stream) ...")
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

    _log(f"  Downloading {dest.name} "
         f"({total / 1e9:.1f} GB, {num_connections} connections) ...")

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


# ---------------------------------------------------------------------------
# Archive extraction — prefer system tools, fallback to parallel Python
# ---------------------------------------------------------------------------

def _extract_tar(archive_path: Path, dest_dir: Path, cleanup: bool = True):
    """Extract tar/tar.gz using system tar (faster) or Python tarfile."""
    if shutil.which("tar"):
        _log(f"  Extracting {archive_path.name} with tar ...")
        subprocess.run(
            ["tar", "-xf", str(archive_path), "-C", str(dest_dir)],
            check=True,
        )
    else:
        _log(f"  Extracting {archive_path.name} with Python ...")
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(str(dest_dir), filter="data")
    _log(f"  Extracted {archive_path.name} → {dest_dir}")
    if cleanup:
        archive_path.unlink(missing_ok=True)


def _extract_zip_members(zip_path, members, dest):
    """Extract a subset of members (for ProcessPoolExecutor)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in members:
            zf.extract(name, dest)


def _extract_zip(archive_path: Path, dest_dir: Path, cleanup: bool = True):
    """Extract zip using system unzip or parallel Python."""
    if shutil.which("unzip"):
        _log(f"  Extracting {archive_path.name} with unzip ...")
        subprocess.run(
            ["unzip", "-q", "-o", str(archive_path), "-d", str(dest_dir)],
            check=True,
        )
    else:
        _log(f"  Extracting {archive_path.name} (parallel Python) ...")
        num_workers = min(os.cpu_count() or 1, 8)
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            dirs = {
                str(dest_dir / os.path.dirname(n))
                for n in names if os.path.dirname(n)
            }
            for d in dirs:
                os.makedirs(d, exist_ok=True)
        chunks = [names[i::num_workers] for i in range(num_workers)]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = [
                pool.submit(_extract_zip_members, str(archive_path), chunk, str(dest_dir))
                for chunk in chunks if chunk
            ]
            for f in as_completed(futs):
                f.result()
    _log(f"  Extracted {archive_path.name} → {dest_dir}")
    if cleanup:
        archive_path.unlink(missing_ok=True)


def _extract_archive(archive_path: Path, dest_dir: Path, cleanup: bool = True):
    """Dispatch to tar or zip extraction."""
    name = archive_path.name
    if name.endswith(".zip"):
        _extract_zip(archive_path, dest_dir, cleanup=cleanup)
    elif name.endswith(".tar") or name.endswith(".tar.gz"):
        _extract_tar(archive_path, dest_dir, cleanup=cleanup)
    else:
        _log(f"  Unknown archive format: {name}, skipping extraction")


# ---------------------------------------------------------------------------
# HuggingFace file resolution — get direct download URL for range requests
# ---------------------------------------------------------------------------

def _hf_resolve_url(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Resolve the direct download URL for a file in a HF repo."""
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


# ---------------------------------------------------------------------------
# Download tasks — each runs in its own thread
# ---------------------------------------------------------------------------

def _download_pangea_json(output_dir: Path, num_connections: int):
    """Download PangeaIns.json (~2 GB) with multi-connection."""
    json_path = output_dir / "PangeaIns.json"
    if json_path.exists():
        _log(f"PangeaIns.json already exists at {json_path}")
        return json_path

    url = _hf_resolve_url(PANGEA_REPO, "PangeaIns.json")
    _fast_download(url, json_path, num_connections=num_connections)

    # Verify
    with open(json_path) as f:
        head = f.read(2000)
    if '"conversations"' in head and '"language"' in head:
        _log("  ✓ PangeaIns.json schema verified (conversations, language)")
    else:
        _log("  ⚠ PangeaIns.json schema may not match expected format")
    return json_path


def _download_aya(cache_dir: str | None):
    """Pre-download Aya Dataset parquet files via HF datasets."""
    from datasets import load_dataset
    _log("Pre-downloading CohereForAI/aya_dataset ...")
    ds = load_dataset(
        "CohereForAI/aya_dataset",
        split="train",
        cache_dir=cache_dir,
    )
    _log(f"  ✓ Aya Dataset: {len(ds)} examples cached")


def _download_and_extract_subset(
    subset: str,
    output_dir: Path,
    num_connections: int,
    keep_archives: bool,
    _repo_files: list[str],
):
    """Download and extract all archives for one Pangea image subset."""
    # Find archive files for this subset
    archive_files = [
        f for f in _repo_files
        if f.startswith(subset + "/") and (
            f.endswith(".tar") or f.endswith(".zip") or f.endswith(".tar.gz")
        )
    ]
    # Skip multi-part files
    archive_files = [f for f in archive_files
                     if not f.endswith((".partaa", ".partab"))]

    if not archive_files:
        _log(f"  [{subset}] No archives found, skipping")
        return

    subset_dir = output_dir / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing = [
        f for f in subset_dir.iterdir()
        if not f.name.endswith((".tar", ".zip", ".tar.gz", ".tmp"))
    ] if subset_dir.exists() else []
    if existing:
        _log(f"  [{subset}] Already extracted ({len(existing)} items), skipping")
        return

    for fname in archive_files:
        local_path = output_dir / fname
        url = _hf_resolve_url(PANGEA_REPO, fname)

        _log(f"  [{subset}] Downloading {Path(fname).name} ...")
        _fast_download(url, local_path, num_connections=num_connections)
        _extract_archive(local_path, subset_dir, cleanup=not keep_archives)

    _log(f"  [{subset}] Done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download multilingual training datasets with parallel connections"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Root directory for downloaded data",
    )
    parser.add_argument(
        "--download-images", action="store_true",
        help="Also download Pangea image archives (large, ~100+ GB total)",
    )
    parser.add_argument(
        "--image-subsets", nargs="*", default=None,
        help="Specific image subsets to download (default: core set)",
    )
    parser.add_argument(
        "--num-connections", type=int, default=8,
        help="Parallel HTTP range-request connections per file (default: 8)",
    )
    parser.add_argument(
        "--max-parallel-sources", type=int, default=4,
        help="Max concurrent source downloads (default: 4)",
    )
    parser.add_argument(
        "--keep-archives", action="store_true",
        help="Keep archive files after extraction",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--skip-aya", action="store_true",
        help="Skip pre-downloading Aya Dataset",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 60)
    _log("Multilingual dataset downloader")
    _log(f"  Output:      {output_dir}")
    _log(f"  Connections: {args.num_connections} per file")
    _log(f"  Parallel:    {args.max_parallel_sources} sources")
    _log(f"  Images:      {'yes' if args.download_images else 'no'}")
    _log("=" * 60)

    # Pre-fetch the repo file list once (avoids repeated API calls)
    repo_files: list[str] = []
    if args.download_images:
        _log("\nFetching file list from neulab/PangeaInstruct ...")
        api = HfApi()
        repo_files = list(api.list_repo_files(
            repo_id=PANGEA_REPO, repo_type="dataset",
        ))
        _log(f"  {len(repo_files)} files in repo")

    # Determine image subsets
    subsets = (args.image_subsets or DEFAULT_IMAGE_SUBSETS) if args.download_images else []

    # Launch all downloads concurrently
    # Each source runs in its own thread; within each thread, the file download
    # uses N parallel HTTP range-request connections.
    max_workers = min(
        args.max_parallel_sources,
        2 + len(subsets),  # JSON + Aya + image subsets
    )

    _log(f"\nLaunching downloads (up to {max_workers} concurrent) ...\n")

    with ThreadPoolExecutor(max_workers=max(max_workers, 1)) as pool:
        futures: dict = {}

        # 1. PangeaIns.json (always)
        futures[pool.submit(
            _download_pangea_json, output_dir, args.num_connections,
        )] = "PangeaIns.json"

        # 2. Aya Dataset (pre-download parquet cache)
        if not args.skip_aya:
            futures[pool.submit(
                _download_aya, args.cache_dir,
            )] = "CohereForAI/aya_dataset"

        # 3. Image subsets (each in its own thread)
        for subset in subsets:
            futures[pool.submit(
                _download_and_extract_subset,
                subset, output_dir, args.num_connections,
                args.keep_archives, repo_files,
            )] = f"images:{subset}"

        # Wait with progress reporting
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            label = futures[fut]
            done += 1
            try:
                fut.result()
                _log(f"✓ [{done}/{total}] Finished {label}")
            except Exception as exc:
                _log(f"✗ [{done}/{total}] Failed {label}: {exc}")

    _log(f"\n{'=' * 60}")
    _log("Download complete!")
    _log(f"  Pangea data: {output_dir}")
    _log("  Aya Dataset: cached by HuggingFace datasets library")
    _log("=" * 60)


if __name__ == "__main__":
    main()
