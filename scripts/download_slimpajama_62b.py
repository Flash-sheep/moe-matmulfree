#!/usr/bin/env python3
"""Download SlimPajama-62B using wget -c with proxy, resumable.

Usage:
    python3 scripts/download_slimpajama_62b.py
    python3 scripts/download_slimpajama_62b.py --split validation  # only validation
    python3 scripts/download_slimpajama_62b.py --split test         # only test
    python3 scripts/download_slimpajama_62b.py --split train        # only train
"""

import subprocess
import sys
import os
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/venketh/SlimPajama-62B/resolve/main"
PROXY = "http://10.0.16.44:39081"
OUTPUT_DIR = Path("/home/storage/yjl/matmulfreellm_assets/datasets/SlimPajama-62B")

FILES = {
    "train": [f"train/chunk2_{i}.jsonl.zst" for i in range(6)],
    "validation": [f"validation/chunk{i}.jsonl.zst" for i in range(1, 6)],
    "test": [f"test/chunk{i}.jsonl.zst" for i in range(1, 6)],
}

ALL_FILES = FILES["train"] + FILES["validation"] + FILES["test"]


def download_file(url, output_path):
    """Download a single file using wget -c (resume) with proxy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget", "-c", "--progress=bar:force:noscroll",
        "-e", f"https_proxy={PROXY}",
        "-O", str(output_path),
        url,
    ]
    print(f"[download] {output_path.name} ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Failed to download {url}")
        return False
    # Verify file exists and has size > 0
    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[download] {output_path.name} OK ({size_mb:.0f} MB)")
        return True
    return False


def main():
    split_filter = None
    if len(sys.argv) > 2 and sys.argv[1] == "--split":
        split_filter = sys.argv[2]
        if split_filter not in FILES:
            print(f"Unknown split: {split_filter}. Use: train, validation, test")
            sys.exit(1)

    if split_filter:
        file_list = FILES[split_filter]
    else:
        file_list = ALL_FILES

    print(f"Target: {OUTPUT_DIR}")
    print(f"Files to download: {len(file_list)}")
    print(f"Proxy: {PROXY}")
    print()

    success = 0
    failed = []
    for i, fpath in enumerate(file_list, 1):
        url = f"{BASE_URL}/{fpath}"
        output = OUTPUT_DIR / fpath
        print(f"[{i}/{len(file_list)}] ", end="")
        if output.exists():
            # Check if already fully downloaded by comparing with server size
            existing_size = output.stat().st_size
            if existing_size > 10 * 1024 * 1024:  # > 10MB, likely complete
                print(f"{fpath} already exists ({existing_size/(1024*1024):.0f} MB), skipping")
                success += 1
                continue
        if download_file(url, output):
            success += 1
        else:
            failed.append(fpath)

    print()
    print(f"Download complete: {success}/{len(file_list)} succeeded")
    if failed:
        print(f"Failed files ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
        print("Re-run to retry (wget -c will resume).")

    # Summary of downloaded sizes
    total_size = 0
    for fpath in file_list:
        p = OUTPUT_DIR / fpath
        if p.exists():
            total_size += p.stat().st_size
    print(f"Total downloaded: {total_size/(1024**3):.1f} GB")


if __name__ == "__main__":
    main()
