#!/usr/bin/env python3
"""Download raw datasets for behavior-lab.

Supports automatic download for SUBTLE, Shank3KO, and MABe22.
Rat7M requires manual download due to size.

Usage:
    python scripts/download_data.py --dataset subtle
    python scripts/download_data.py --dataset shank3ko
    python scripts/download_data.py --dataset mabe22
    python scripts/download_data.py --dataset rat7m      # prints instructions
    python scripts/download_data.py --all                # subtle+shank3ko+mabe22
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return

    label = desc or dest.name
    print(f"  Downloading {label}...")

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    {pct:5.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r    {mb:.1f} MB downloaded")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
        # Remove empty files (failed downloads that wrote 0 bytes)
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
            raise RuntimeError("Downloaded file is empty")
        print(f"\n  Saved: {dest}")
    except Exception:
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        raise


# =============================================================================
# SUBTLE: GitHub CSV files from jeakwon/subtle
# =============================================================================

# Two batches: y5a5 (10 files) and y3a6 (9 files)
# Located at dataset/{batch}/coords/{name}.csv
SUBTLE_BASE_URL = "https://raw.githubusercontent.com/jeakwon/subtle/main/dataset"
SUBTLE_FILES = {
    "y5a5": [
        "adult_6112.csv", "adult_6115.csv", "adult_6116.csv",
        "adult_6127.csv", "adult_7678.csv",
        "young_7100.csv", "young_7678.csv", "young_8294.csv",
        "young_8296.csv", "young_8301.csv",
    ],
    "y3a6": [
        "adult_8294.csv", "adult_8296.csv", "adult_8301.csv",
        "adult_8765.csv", "adult_8767.csv", "adult_8789.csv",
        "young_8765.csv", "young_8767.csv", "young_8789.csv",
    ],
}


def download_subtle() -> None:
    """Download SUBTLE CSV files from GitHub."""
    print("\n" + "=" * 50)
    print("SUBTLE: Mouse Spontaneous Behavior (3D)")
    print("=" * 50)

    out_dir = RAW_DIR / "subtle"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in SUBTLE_FILES.values())
    downloaded = 0

    for batch, files in SUBTLE_FILES.items():
        for fname in files:
            url = f"{SUBTLE_BASE_URL}/{batch}/coords/{fname}"
            dest = out_dir / f"{batch}_{fname}"
            try:
                _download_file(url, dest, f"{fname} ({downloaded+1}/{total})")
                downloaded += 1
            except Exception as e:
                print(f"  Warning: Failed to download {fname}: {e}")

    print(f"\nSUBTLE: {downloaded}/{total} files downloaded to {out_dir}")


# =============================================================================
# Shank3KO: Zenodo .mat file
# =============================================================================

SHANK3KO_URL = "https://zenodo.org/records/4629544/files/Shank3KO_mice_slk3D.mat"
SHANK3KO_SIZE_MB = 193.8


def download_shank3ko() -> None:
    """Download Shank3KO .mat file from Zenodo."""
    print("\n" + "=" * 50)
    print(f"Shank3KO: Knockout Mouse Behavior (~{SHANK3KO_SIZE_MB} MB)")
    print("=" * 50)

    out_dir = RAW_DIR / "shank3ko"
    dest = out_dir / "Shank3KO_mice_slk3D.mat"
    _download_file(SHANK3KO_URL, dest)


# =============================================================================
# MABe22: Caltech mouse triplet task
# =============================================================================

MABE22_BASE_URL = "https://data.caltech.edu/records/rdsa8-rde65/files"
MABE22_FILES = [
    "mouse_user_train.npy",       # 417.8 MB — pose keypoints (training)
    "mouse_sample_submission.npy", # 65.9 MB — sample submission format
]


def download_mabe22() -> None:
    """Download MABe2022 mouse triplet data from Caltech."""
    print("\n" + "=" * 50)
    print("MABe22: Mouse Triplet Behavior")
    print("=" * 50)

    out_dir = RAW_DIR / "mabe22"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in MABE22_FILES:
        url = f"{MABE22_BASE_URL}/{fname}"
        dest = out_dir / fname
        try:
            _download_file(url, dest)
        except Exception as e:
            print(f"  Warning: Failed to download {fname}: {e}")
            print(f"  Manual URL: {url}")


# =============================================================================
# Rat7M: Manual download instructions
# =============================================================================

def download_rat7m() -> None:
    """Print Rat7M download instructions (too large for auto-download)."""
    print("\n" + "=" * 50)
    print("Rat7M: 3D Rat Motion Capture")
    print("=" * 50)
    print()
    print("Rat7M data is too large for automatic download (several GB).")
    print("Please download manually from Figshare:")
    print()
    print("  Collection: https://springernature.figshare.com/collections/5295370")
    print()
    print("Download the .mat files and place them in:")
    print(f"  {RAW_DIR / 'rat7m'}/")
    print()
    print("Expected files:")
    print("  - subject1_session1.mat")
    print("  - subject1_session2.mat")
    print("  - ...")
    print()
    print("Each .mat file should contain 'mocap_markers' with shape (T, 20, 3)")


# =============================================================================
# CLI
# =============================================================================

DATASETS = {
    "subtle": download_subtle,
    "shank3ko": download_shank3ko,
    "mabe22": download_mabe22,
    "rat7m": download_rat7m,
}

AUTO_DATASETS = ["subtle", "shank3ko", "mabe22"]


def main():
    parser = argparse.ArgumentParser(
        description="Download raw datasets for behavior-lab"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all auto-downloadable datasets (subtle, shank3ko, mabe22)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR})",
    )
    args = parser.parse_args()

    if args.all:
        for name in AUTO_DATASETS:
            DATASETS[name]()
        print("\n" + "=" * 50)
        print("All auto-downloadable datasets complete.")
        print("Run `python scripts/preprocess_data.py --all` to preprocess.")
    elif args.dataset:
        DATASETS[args.dataset]()
    else:
        parser.print_help()
        print("\nAvailable datasets:")
        for name in DATASETS:
            auto = "auto" if name in AUTO_DATASETS else "manual"
            print(f"  {name:12s} ({auto})")


if __name__ == "__main__":
    main()
