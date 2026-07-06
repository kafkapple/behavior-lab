"""Multi-seed stability check for the CalMS21 clustering ARI benchmark.

The existing 6-model comparison (docs/behavior_lab_technical_notes.md §9) reports
a single run per model/dataset. KMeans/UMAP are stochastic, so a single seed can't
tell a real effect from seed noise. This wraps the same pipeline
(experiments.discovery.compare_discovery_methods -> models.discovery.clustering)
in a seed loop and reports ARI/silhouette as mean +/- std.

Usage:
    python scripts/run_discovery_multiseed.py
"""
import os
# UMAP+numba segfaults under multi-threaded OMP on this env (exit 139); force
# single-threaded BLAS/OMP before numpy/umap import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys; sys.path.insert(0, "src")

import numpy as np

from behavior_lab.data.loaders import get_loader
from behavior_lab.experiments.discovery import compare_discovery_methods
from behavior_lab.evaluation import compute_cluster_metrics

SEEDS = [0, 42, 123, 456, 789]
MAX_FRAMES = 8000


def load_calms21(max_frames: int = MAX_FRAMES, seed: int = 0):
    """Concatenate CalMS21 train sequences into one (T,K,D) array + per-frame GT.

    Subsamples whole sequences class-stratified (by each sequence's single action
    label) rather than truncating to the first N frames -- a plain prefix
    truncation isn't guaranteed class-balanced since sequences aren't pre-shuffled
    by label (260706 research-note Limitation item).
    """
    loader = get_loader("calms21", data_dir="data/calms21", skeleton_name="calms21")
    sequences = loader.load_split("train")
    if max_frames:
        frames_per_seq = sequences[0].keypoints.shape[0]
        n_seq_needed = max(1, max_frames // frames_per_seq)
        labels = np.array([s.metadata["action_label"] for s in sequences])
        classes = np.unique(labels)
        per_class = max(1, n_seq_needed // len(classes))
        rng = np.random.default_rng(seed)
        idx = np.concatenate([
            rng.choice(np.flatnonzero(labels == c), size=min(per_class, int((labels == c).sum())), replace=False)
            for c in classes
        ])
        rng.shuffle(idx)
        sequences = [sequences[i] for i in idx]
    kp = np.concatenate([s.keypoints for s in sequences], axis=0)
    gt = np.concatenate([s.labels for s in sequences], axis=0)
    return kp.astype(np.float32), gt


def run_seed(kp: np.ndarray, gt: np.ndarray, seed: int) -> dict:
    run = compare_discovery_methods(
        kp, methods=["kmeans_pca_umap"], n_clusters=4, random_state=seed,
    )[0]
    m = compute_cluster_metrics(run.result.features, run.result.labels, gt)
    return {"ari": m.ari, "nmi": m.nmi, "silhouette": m.silhouette}


def main():
    kp, gt = load_calms21()
    print(f"CalMS21: {kp.shape[0]} frames, {kp.shape[1]} keypoints, "
          f"{len(np.unique(gt))} GT classes")

    rows = [run_seed(kp, gt, seed) for seed in SEEDS]
    for seed, row in zip(SEEDS, rows):
        print(f"  seed={seed:>4}  ARI={row['ari']:.4f}  NMI={row['nmi']:.4f}  "
              f"silhouette={row['silhouette']:.4f}")

    ari = np.array([r["ari"] for r in rows])
    nmi = np.array([r["nmi"] for r in rows])
    sil = np.array([r["silhouette"] for r in rows])
    print(f"\n| Metric | Mean | Std | N seeds |")
    print(f"|---|---:|---:|---:|")
    print(f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {len(SEEDS)} |")
    print(f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {len(SEEDS)} |")
    print(f"| Silhouette | {sil.mean():.4f} | {sil.std():.4f} | {len(SEEDS)} |")


if __name__ == "__main__":
    main()
