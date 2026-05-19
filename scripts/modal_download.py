"""
Download LLaVA-Pretrain to a Modal volume.

The volume has a 500k-inode quota; LLaVA-Pretrain has 558k images, so we keep
images.zip on the volume and read members directly at training time (see
pipeline/data.py:AlignmentDataset). The ::extract entrypoint is retained for
contexts without that inode constraint.

Usage:
    modal run scripts/modal_download.py                          # download zip + json only
    modal run scripts/modal_download.py::extract                 # extract zip (needs >558k inodes)
    modal run scripts/modal_download.py::cleanup_partial_extract # free inodes from a half-done extract
"""

from pathlib import Path

import modal

app = modal.App("tayavision-download")
volume = modal.Volume.from_name("tayavision-data")

image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")

DATA_DIR = "/data/llava-pretrain"
# LLaVA-Pretrain images.zip contains shards 00000/ through 00659/.
# Presence of the last shard is a cheap proxy for a complete extraction.
LAST_SHARD = "00659"


def _is_extracted(output) -> bool:
    last = output / LAST_SHARD
    return last.exists() and any(last.iterdir())


@app.function(image=image, volumes={"/data": volume}, timeout=14400, ephemeral_disk=524_288)
def download():
    """Download the JSON and images.zip onto the volume.

    Does NOT extract — training reads images directly from the zip via
    AlignmentDataset (see pipeline/data.py). Extraction would need 558k+
    inodes on the volume.
    """
    import json
    import os
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    os.environ["HF_HUB_CACHE"] = "/data/.hf_cache"

    output = Path(DATA_DIR)
    output.mkdir(parents=True, exist_ok=True)
    zip_path = output / "images.zip"
    json_path = output / "blip_laion_cc_sbu_558k.json"

    if zip_path.exists() and json_path.exists():
        print(f"Both {zip_path.name} and {json_path.name} already on volume, skipping.")
        return

    print("Downloading conversations JSON...")
    downloaded_json = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="blip_laion_cc_sbu_558k.json",
        repo_type="dataset",
        local_dir=str(output),
    )
    with open(downloaded_json) as f:
        convos = json.load(f)
    print(f"  {len(convos)} conversations")

    print("Downloading images.zip (~13GB, this will take a while)...")
    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="images.zip",
        repo_type="dataset",
        local_dir=str(output),
    )

    volume.commit()
    print("Done.")


@app.function(image=image, volumes={"/data": volume}, timeout=3600)
def cleanup_partial_extract():
    """Delete numeric shard dirs left over from a half-done extraction.

    Frees inodes on the volume. Safe to run any time — the dataset reads
    from images.zip directly, not from extracted shards.
    """
    import shutil

    output = Path(DATA_DIR)
    if not output.exists():
        print(f"{output} does not exist; nothing to clean up.")
        return

    removed = 0
    for child in output.iterdir():
        if child.is_dir() and child.name.isdigit() and len(child.name) == 5:
            shutil.rmtree(child)
            removed += 1
            if removed % 50 == 0:
                print(f"  removed {removed} shards...")
                volume.commit()

    print(f"Removed {removed} shard directories.")
    volume.commit()
    print("Done.")


def _du_gb(path: Path) -> float:
    """Return total size of `path` in GB, or 0.0 if it doesn't exist."""
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 1e9


@app.function(image=image, volumes={"/data": volume}, timeout=14400, ephemeral_disk=524_288)
def extract():
    """Re-extract images.zip from the volume without re-downloading.

    Iterates member-by-member so a partially-extracted volume can resume
    cheaply (existing files are skipped) and so any failure reports the
    offending member path. Commits the volume periodically so partial
    progress survives an interruption. Purges HF download caches up front
    to free volume space — they're redundant once images.zip is on disk.
    """
    import shutil
    import zipfile
    from pathlib import Path

    output = Path(DATA_DIR)
    zip_path = output / "images.zip"

    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found on volume. Run `modal run scripts/modal_download.py` first."
        )

    cache_paths = [Path("/data/.hf_cache"), output / ".cache"]
    print("Volume usage before cleanup:")
    for p in [output, *cache_paths, zip_path]:
        print(f"  {p}: {_du_gb(p):.2f} GB")

    for cache in cache_paths:
        if cache.exists():
            print(f"Removing {cache} ...")
            shutil.rmtree(cache)

    volume.commit()
    print(f"Free space (ephemeral): {shutil.disk_usage(output).free / 1e9:.1f} GB")

    print(f"Extracting {zip_path} → {output} ...")
    skipped = 0
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        for i, member in enumerate(members):
            target = output / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == member.file_size:
                skipped += 1
            else:
                try:
                    zf.extract(member, str(output))
                    extracted += 1
                except OSError as e:
                    raise OSError(
                        f"Failed extracting {member.filename} (member {i}/{total}): {e}"
                    ) from e
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{total}  extracted={extracted}  skipped={skipped}")
                volume.commit()

    print(f"Extraction complete. extracted={extracted}  skipped={skipped}  total={total}")

    if _is_extracted(output):
        zip_path.unlink()
        print("Deleted images.zip to save space.")
    else:
        print(f"Warning: shard {LAST_SHARD}/ missing after extract; leaving images.zip in place.")

    volume.commit()
    print("Done.")


@app.local_entrypoint()
def main():
    download.remote()
