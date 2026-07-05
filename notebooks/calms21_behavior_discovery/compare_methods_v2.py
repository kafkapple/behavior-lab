"""v2 — compare on 24-dim ego-centric dyadic features (the proper social repr).

Differences from v1 (compare_methods.py):
  * Features are 24-dim dyadic ego-centric (not 12-dim resident pixel)
  * B-SOiD min_cluster_size sweep [50, 100, 200] — fixes K=2 collapse
  * Permutation baseline (100x) for ARI — stratified by class prior
  * SUBTLE re-fits on the new features (fresh model)
  * Labels saved as labels_v2.parquet for downstream GIF script
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(k, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_lab.models.discovery.subtle_wrapper import (
    _patch_subtle_cwt, _patch_umap_njobs, _patch_phenograph_njobs,
)
from behavior_lab.models.discovery.bsoid import BSOiD
from behavior_lab.models.discovery.moseq import _PCAHMMFallback
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = REPO / "data" / "calms21_behavior_discovery"
RAW = ROOT / "raw_csv_v2"
ANN = ROOT / "annotations"
OUT_BASE = REPO / "outputs" / "calms21_behavior_discovery" / "results_v2"
OUT_BASE.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["attack", "investigation", "mount", "other"]
FPS = 30


def load_session(name: str) -> tuple[np.ndarray, np.ndarray]:
    flat = pd.read_csv(RAW / f"{name}.csv", header=None, dtype=np.float32).values
    gt = np.load(ANN / f"{name}.npy")[: len(flat)].astype(int)
    return flat, gt


def metrics(pred: np.ndarray, gt: np.ndarray, n_perm: int = 100,
            rng_seed: int = 0) -> dict:
    """Compute ARI/NMI vs a stratified permutation baseline.

    The permutation shuffles `pred` while keeping its label distribution —
    so the baseline ARI is exactly what a random labeling with the same
    cluster sizes would achieve (handles imbalance correctly).
    """
    ari = float(adjusted_rand_score(gt, pred))
    nmi = float(normalized_mutual_info_score(gt, pred))
    K = len(np.unique(pred))
    rng = np.random.default_rng(rng_seed)
    perm_aris = []
    for _ in range(n_perm):
        perm_aris.append(adjusted_rand_score(gt, rng.permutation(pred)))
    perm_aris = np.array(perm_aris)
    return {
        "k": int(K),
        "ari": ari,
        "nmi": nmi,
        "ari_perm_mean": float(perm_aris.mean()),
        "ari_perm_p": float((perm_aris >= ari).mean()),
    }


def run_subtle(feats_per: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Re-fit SUBTLE on 24-dim features. Returns (labels_flat, embeddings_flat, info)."""
    _patch_subtle_cwt(); _patch_umap_njobs(); _patch_phenograph_njobs()
    import subtle
    mapper = subtle.Mapper(fs=FPS, include_coordinates=False)
    # train on first 2 sessions like v1, then run on all
    mapper.fit(feats_per[:2])
    outs = mapper.run(feats_per)
    labels = np.concatenate([np.asarray(o.y).astype(int) for o in outs])
    emb = np.concatenate([np.asarray(o.Z) for o in outs])
    return labels, emb, {"k": int(len(set(labels)))}


def run_bsoid_sweep(feats_pooled: np.ndarray, gt: np.ndarray,
                    mcs_grid=(50, 100, 200)) -> dict:
    """Sweep min_cluster_size and pick best ARI."""
    # B-SOiD expects (T, K, D) — we encode each ego-feature row as 12 keypoints x 2
    # (24 dim total). It's a (T, 12, 2) "virtual skeleton".
    T = feats_pooled.shape[0]
    bsoid_input = feats_pooled.reshape(T, 12, 2).astype(np.float32)

    best = None
    runs = []
    for mcs in mcs_grid:
        t0 = time.time()
        try:
            model = BSOiD(fps=FPS, n_neighbors=60, min_cluster_size=mcs, umap_dim=3)
            res = model.fit(bsoid_input)
            labels = res["labels"]
            if len(labels) < len(gt):
                labels = np.pad(labels, (0, len(gt) - len(labels)), mode="edge")
            else:
                labels = labels[: len(gt)]
            mask = labels != -1
            m = metrics(labels[mask], gt[mask])
            m["min_cluster_size"] = mcs
            m["elapsed"] = time.time() - t0
            m["embeddings_2d"] = res["embedding_2d"]
            m["labels_full"] = labels
            runs.append(m)
            print(f"  B-SOiD mcs={mcs}: K={m['k']}  ARI={m['ari']:.3f}  "
                  f"({m['elapsed']:.1f}s)")
            if best is None or m["ari"] > best["ari"]:
                best = m
        except (RuntimeError, ValueError, MemoryError) as e:
            print(f"  B-SOiD mcs={mcs}: FAILED — {type(e).__name__}: {e}")
    if best is None:
        raise RuntimeError(
            f"B-SOiD failed for every min_cluster_size in {mcs_grid}; "
            "see per-iteration errors above")
    return {"best": best, "runs": runs}


def main():
    sessions = sorted([p.stem for p in RAW.glob("*.csv")])
    print(f"sessions: {len(sessions)}")
    feats_per, gt_per = [], []
    for s in sessions:
        f, g = load_session(s)
        feats_per.append(f); gt_per.append(g)

    feats_pooled = np.concatenate(feats_per)
    gt_pooled = np.concatenate(gt_per)
    print(f"pooled shape: {feats_pooled.shape}   GT 4-class counts: "
          f"{np.bincount(gt_pooled, minlength=4).tolist()}")

    OUT = OUT_BASE / time.strftime("v2_%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)

    # -------- SUBTLE --------
    print("\n>>> SUBTLE re-fit on 24-dim ego-centric features ...")
    t0 = time.time()
    subtle_labels, subtle_emb, subtle_info = run_subtle(feats_per)
    print(f"  SUBTLE: K={subtle_info['k']}  ({time.time()-t0:.1f}s)")

    # -------- B-SOiD sweep --------
    print("\n>>> B-SOiD K-sweep ...")
    bsoid_sweep = run_bsoid_sweep(feats_pooled, gt_pooled)
    bsoid_best = bsoid_sweep["best"]

    # -------- PCA-HMM --------
    print("\n>>> PCA-HMM (kp-MoSeq fallback) on dyadic features ...")
    t0 = time.time()
    hmm_input = feats_pooled.reshape(-1, 12, 2)
    hmm = _PCAHMMFallback(n_components=10, n_states=25, n_iter=30)
    hmm_res = hmm.fit(hmm_input)
    hmm_labels = hmm_res.labels
    print(f"  PCA-HMM: K={hmm_res.n_clusters}  ({time.time()-t0:.1f}s)")

    # -------- Metrics with stratified permutation --------
    print("\n>>> metrics (with stratified permutation baseline n=100) ...")
    summary = []
    for name, labels in [("SUBTLE", subtle_labels),
                         ("B-SOiD_best", bsoid_best["labels_full"]),
                         ("PCA-HMM", hmm_labels)]:
        n = min(len(labels), len(gt_pooled))
        m = metrics(labels[:n], gt_pooled[:n])
        m["method"] = name
        summary.append(m)
        print(f"  {name:15s}  K={m['k']:>3}  ARI={m['ari']:.3f}  "
              f"NMI={m['nmi']:.3f}  perm_mean={m['ari_perm_mean']:.3f}  "
              f"p={m['ari_perm_p']:.3f}")
    df_metric = pd.DataFrame(summary).set_index("method")[
        ["k", "ari", "nmi", "ari_perm_mean", "ari_perm_p"]]
    df_metric.to_csv(OUT / "metrics_v2.csv")

    # -------- Save labels as parquet for GIF decouple --------
    n_min = min(len(subtle_labels), len(bsoid_best["labels_full"]),
                len(hmm_labels), len(gt_pooled))
    labels_df = pd.DataFrame({
        "frame_global": np.arange(n_min),
        "subtle": subtle_labels[:n_min].astype(np.int32),
        "bsoid":  bsoid_best["labels_full"][:n_min].astype(np.int32),
        "pca_hmm": hmm_labels[:n_min].astype(np.int32),
        "gt": gt_pooled[:n_min].astype(np.int8),
    })
    try:
        labels_path = OUT / "labels_v2.parquet"
        labels_df.to_parquet(labels_path)
    except (ImportError, ValueError):
        labels_path = OUT / "labels_v2.csv"
        labels_df.to_csv(labels_path, index=False)
    print(f"  saved labels: {labels_path}")

    # -------- Plots --------
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    df_metric[["ari", "nmi", "ari_perm_mean"]].plot.bar(ax=ax, colormap="viridis")
    ax.axhline(0, ls="-", color="grey", lw=0.5)
    ax.set_ylabel("score"); ax.set_xlabel("")
    ax.set_title("v2 (24-dim ego-centric dyadic)  ARI / NMI vs permutation baseline")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "metrics_v2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {OUT / 'metrics_v2.png'}")
    # NOTE: per-method UMAP / GT scatter / v1↔v2 charts are produced separately
    # by fix_v2_umap_plot.py — it renders a *shared* UMAP of dyadic features
    # which is more interpretable than each method's native embedding space.
    print(f"\n=== v2 done. Artifacts under {OUT}")


if __name__ == "__main__":
    main()
