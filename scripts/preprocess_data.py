#!/usr/bin/env python3
"""Preprocess raw datasets into standardized .npz format.

Converts raw data (CSV, .mat, .npy) into uniform (T, K, D) .npz files
with optional preprocessing (interpolation, smoothing, normalization).

Usage:
    python scripts/preprocess_data.py --dataset subtle
    python scripts/preprocess_data.py --dataset shank3ko
    python scripts/preprocess_data.py --dataset mabe22
    python scripts/preprocess_data.py --dataset rat7m
    python scripts/preprocess_data.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR = ROOT / "data" / "raw"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"


def preprocess_subtle() -> None:
    """Convert SUBTLE CSV files to .npz (T, 9, 3)."""
    print("\n" + "=" * 50)
    print("Preprocessing SUBTLE")
    print("=" * 50)

    raw_dir = RAW_DIR / "subtle"
    out_dir = PREPROCESSED_DIR / "subtle"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"  No CSV files found in {raw_dir}")
        return

    all_keypoints = []
    for csv_path in csv_files:
        try:
            # SUBTLE CSV: headerless, 27 columns (9 joints × 3D)
            data = np.loadtxt(csv_path, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)

            n_cols = data.shape[1]
            if n_cols >= 27:
                keypoints = data[:, :27].reshape(-1, 9, 3)
            else:
                print(f"  Warning: {csv_path.name} has {n_cols} columns, expected >=27")
                continue

            all_keypoints.append(keypoints.astype(np.float32))
            print(f"  {csv_path.name}: {keypoints.shape}")
        except Exception as e:
            print(f"  Warning: Failed to load {csv_path.name}: {e}")

    if not all_keypoints:
        print("  No valid data loaded")
        return

    # Save individual files
    for i, (kp, csv_path) in enumerate(zip(all_keypoints, csv_files)):
        out_path = out_dir / f"{csv_path.stem}.npz"
        np.savez_compressed(out_path, keypoints=kp)
        print(f"  Saved: {out_path.name} {kp.shape}")

    # Save concatenated
    combined = np.concatenate(all_keypoints, axis=0)
    combined_path = out_dir / "subtle_all.npz"
    np.savez_compressed(combined_path, keypoints=combined)
    print(f"  Combined: {combined_path.name} {combined.shape}")


def preprocess_shank3ko() -> None:
    """Convert Shank3KO .mat to .npz (T, 16, 3).

    Shank3KO_mice_slk3D.mat structure:
      mice_slk3D: (1, N_mice) structured array with fields:
        CoordX, CoordY, CoordZ: each (T, 16) — 27000 frames × 16 joints
        Genotypes: 'KO' or 'WT'
        Rec_name: recording identifier
    """
    print("\n" + "=" * 50)
    print("Preprocessing Shank3KO")
    print("=" * 50)

    raw_dir = RAW_DIR / "shank3ko"
    out_dir = PREPROCESSED_DIR / "shank3ko"
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(raw_dir.glob("*.mat"))
    if not mat_files:
        print(f"  No .mat files found in {raw_dir}")
        return

    from scipy.io import loadmat

    for mat_path in mat_files:
        try:
            mat = loadmat(str(mat_path))
            available_keys = [k for k in mat if not k.startswith("_")]
            print(f"  {mat_path.name} keys: {available_keys}")

            if "mice_slk3D" in mat:
                # Structured array with per-mouse recordings
                data = mat["mice_slk3D"]
                n_mice = data.shape[1]
                print(f"  Found {n_mice} mice recordings")

                for i in range(n_mice):
                    rec = data[0, i]
                    cx = np.array(rec["CoordX"], dtype=np.float32)  # (T, 16)
                    cy = np.array(rec["CoordY"], dtype=np.float32)
                    cz = np.array(rec["CoordZ"], dtype=np.float32)
                    # Stack to (T, 16, 3)
                    keypoints = np.stack([cx, cy, cz], axis=-1)

                    genotype = str(rec["Genotypes"].flatten()[0])
                    rec_name = str(rec["Rec_name"].flatten()[0])
                    safe_name = rec_name.replace("-", "_")

                    out_path = out_dir / f"{safe_name}.npz"
                    np.savez_compressed(
                        out_path, keypoints=keypoints, genotype=genotype
                    )
                    if i < 3 or i == n_mice - 1:
                        print(f"  [{i+1}/{n_mice}] {rec_name}: {keypoints.shape} ({genotype})")
                    elif i == 3:
                        print(f"  ... ({n_mice - 4} more)")

                print(f"  Total: {n_mice} recordings saved to {out_dir}")
            else:
                print(f"  Warning: No 'mice_slk3D' key found. Keys: {available_keys}")

        except Exception as e:
            print(f"  Warning: Failed to process {mat_path.name}: {e}")


def preprocess_mabe22() -> None:
    """Convert MABe22 .npy to .npz with proper multi-animal shape.

    MABe22 data formats:
      mouse_user_train.npy: dict with 'sequences' → {id: {keypoints: (T, 3, 12, 2), annotations: (2, T)}}
      mouse_triplet_{train,test}.npy: (N, T, 3, 12, 2) or (N, T, 36, 2)

    Output: (N, T, 36, 2) per file — 3 mice × 12 joints flattened.
    """
    print("\n" + "=" * 50)
    print("Preprocessing MABe22")
    print("=" * 50)

    raw_dir = RAW_DIR / "mabe22"
    out_dir = PREPROCESSED_DIR / "mabe22"
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(raw_dir.glob("*.npy"))
    if not npy_files:
        print(f"  No .npy files found in {raw_dir}")
        return

    for npy_path in npy_files:
        try:
            data = np.load(npy_path, allow_pickle=True)
            print(f"  {npy_path.name}: shape={data.shape}, dtype={data.dtype}")

            # Case 1: Object array wrapping a dict (e.g., mouse_user_train.npy)
            if data.shape == () and data.dtype == object:
                obj = data.item()
                if isinstance(obj, dict) and "sequences" in obj:
                    seqs_dict = obj["sequences"]
                    vocab = obj.get("vocabulary", [])
                    print(f"    Dict format: {len(seqs_dict)} sequences, vocab={vocab}")

                    all_kp = []
                    all_ann = []
                    for seq_id, seq_data in seqs_dict.items():
                        kp = seq_data["keypoints"]  # (T, 3, 12, 2)
                        T, M, K, D = kp.shape
                        kp_flat = kp.reshape(T, M * K, D)  # (T, 36, 2)
                        all_kp.append(kp_flat.astype(np.float32))
                        if "annotations" in seq_data:
                            all_ann.append(seq_data["annotations"])

                    keypoints = np.array(all_kp)  # (N, T, 36, 2)
                    out_path = out_dir / f"{npy_path.stem}.npz"
                    save_dict = {"keypoints": keypoints}
                    if all_ann:
                        save_dict["annotations"] = np.array(all_ann)
                    if vocab:
                        save_dict["vocabulary"] = np.array(vocab)
                    np.savez_compressed(out_path, **save_dict)
                    print(f"    Saved: {out_path.name} keypoints={keypoints.shape}")
                    continue
                else:
                    print(f"    Warning: Unexpected dict structure")
                    continue

            # Case 2: Standard array formats
            if data.ndim == 4:
                N, T, K, D = data.shape
                keypoints = data.astype(np.float32)
            elif data.ndim == 5:
                N, T, M, K, D = data.shape
                keypoints = data.reshape(N, T, M * K, D).astype(np.float32)
            elif data.ndim == 3:
                keypoints = data[np.newaxis].astype(np.float32)
            elif data.ndim == 2:
                # Submission format: (total_frames, 5) — skip
                print(f"    Skipping submission format: {data.shape}")
                continue
            else:
                print(f"    Warning: Unexpected shape {data.shape}")
                continue

            out_path = out_dir / f"{npy_path.stem}.npz"
            np.savez_compressed(out_path, keypoints=keypoints)
            print(f"    Saved: {out_path.name} shape={keypoints.shape}")

        except Exception as e:
            print(f"  Warning: Failed to process {npy_path.name}: {e}")


def preprocess_rat7m() -> None:
    """Convert Rat7M .mat/.h5 to .npz (T, 20, 3)."""
    print("\n" + "=" * 50)
    print("Preprocessing Rat7M")
    print("=" * 50)

    raw_dir = RAW_DIR / "rat7m"
    out_dir = PREPROCESSED_DIR / "rat7m"
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(raw_dir.glob("*.mat"))
    h5_files = sorted(raw_dir.glob("*.h5")) + sorted(raw_dir.glob("*.hdf5"))
    all_files = mat_files + h5_files

    if not all_files:
        print(f"  No .mat/.h5 files found in {raw_dir}")
        print("  Run: python scripts/download_data.py --dataset rat7m")
        return

    for fpath in all_files:
        try:
            if fpath.suffix == ".mat":
                from scipy.io import loadmat
                mat = loadmat(str(fpath))
                keypoints = None
                for key in ("mocap_markers", "positions_3d", "keypoints", "data"):
                    if key in mat:
                        keypoints = np.array(mat[key], dtype=np.float32)
                        break
                if keypoints is None:
                    available = [k for k in mat if not k.startswith("_")]
                    print(f"  Warning: No recognized key in {fpath.name}. Found: {available}")
                    continue
            else:
                import h5py
                with h5py.File(fpath, "r") as f:
                    keypoints = None
                    for key in ("positions", "keypoints", "predictions"):
                        if key in f:
                            keypoints = np.array(f[key], dtype=np.float32)
                            break
                    if keypoints is None:
                        print(f"  Warning: No recognized key in {fpath.name}")
                        continue

            if keypoints.ndim == 2:
                T = keypoints.shape[0]
                keypoints = keypoints.reshape(T, -1, 3)

            out_path = out_dir / f"{fpath.stem}.npz"
            np.savez_compressed(out_path, keypoints=keypoints)
            print(f"  Saved: {out_path.name} {keypoints.shape}")

        except Exception as e:
            print(f"  Warning: Failed to process {fpath.name}: {e}")


# =============================================================================
# CLI
# =============================================================================

DATASETS = {
    "subtle": preprocess_subtle,
    "shank3ko": preprocess_shank3ko,
    "mabe22": preprocess_mabe22,
    "rat7m": preprocess_rat7m,
}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw datasets into standardized .npz format"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Preprocess all datasets",
    )
    args = parser.parse_args()

    if args.all:
        for name, func in DATASETS.items():
            func()
        print("\n" + "=" * 50)
        print("All preprocessing complete.")
    elif args.dataset:
        DATASETS[args.dataset]()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
