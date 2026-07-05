"""Compare three unsupervised behavior-discovery methods on CalMS21.

Methods (all from behavior_lab.models.discovery):
  - SUBTLE   — Morlet spectrogram + UMAP + PhenoGraph + DIB (Kwon et al. 2024)
  - B-SOiD   — displacement/distance/angle features + UMAP + HDBSCAN + RF
               (Hsu & Yttri 2021)
  - PCA-HMM  — keypoint-moseq fallback: PCA + Gaussian HMM (proxy for kp-MoSeq
               syllables when the full kpmoseq stack isn't installed)

For each method we report:
  - cluster count K
  - frame-wise weighted purity vs the 4 CalMS21 behavior labels
    (attack / investigation / mount / other)
  - chance-corrected metrics: Adjusted Rand Index (ARI) and Normalized
    Mutual Information (NMI) — both treat 'other' baseline fairly
  - per-method UMAP map coloured by predicted cluster
  - per-method UMAP map coloured by CalMS21 GT (reference)
  - metric bar chart with chance-corrected baseline

Output: PNGs + CSV under data/calms21_behavior_discovery/results/<model_run>/compare_*
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(k, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_lab.models.discovery.bsoid import BSOiD
from behavior_lab.models.discovery.moseq import _PCAHMMFallback

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = REPO / "data" / "calms21_behavior_discovery"
RAW = ROOT / "raw_csv"
ANN = ROOT / "annotations"
RESULTS = REPO / "outputs" / "calms21_behavior_discovery" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["attack", "investigation", "mount", "other"]
FPS = 30


def load_kp(name: str) -> np.ndarray:
    flat = pd.read_csv(RAW / f"{name}.csv", header=None, dtype=np.float32).values
    return flat.reshape(-1, 6, 2)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    K = len(np.unique(pred))
    C = max(int(gt.max()) + 1, 1)
    conf = np.zeros((K, C), dtype=np.int64)
    label_map = {lab: i for i, lab in enumerate(np.unique(pred))}
    for p, g in zip(pred, gt):
        conf[label_map[p], int(g)] += 1
    weights = conf.sum(axis=1) / max(conf.sum(), 1)
    row = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    purity = float((weights * row.max(axis=1)).sum())
    return {
        "k": int(K),
        "purity": purity,
        "ari": float(adjusted_rand_score(gt, pred)),
        "nmi": float(normalized_mutual_info_score(gt, pred)),
    }


def umap_2d(features_or_emb: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """If input is already 2D use as-is, otherwise UMAP to 2D for plotting."""
    if features_or_emb.shape[1] == 2:
        return features_or_emb
    from umap import UMAP
    return UMAP(n_components=2, random_state=seed, n_jobs=1,
                n_neighbors=30).fit_transform(features_or_emb)


def main():
    runs = sorted([p for p in RESULTS.iterdir() if p.is_dir()])
    assert runs, f"no SUBTLE run under {RESULTS}; run the notebook first"
    run = runs[-1]
    print(f"using SUBTLE run: {run.name}")

    sessions = sorted([p.name for p in run.iterdir() if p.is_dir()])
    print(f"sessions: {len(sessions)}")

    # -------- 1. Pool data --------
    kp_per_session = [load_kp(n) for n in sessions]
    gt_per_session = [np.load(ANN / f"{n}.npy")[: len(k)]
                      for n, k in zip(sessions, kp_per_session)]
    lengths = [len(k) for k in kp_per_session]

    kp_all = np.concatenate(kp_per_session)         # (T_all, 6, 2)
    gt_all = np.concatenate(gt_per_session).astype(int)
    print(f"pooled (T,K,D): {kp_all.shape}   GT classes: {np.unique(gt_all).tolist()}")

    # -------- 2. SUBTLE labels (already computed) --------
    subtle_labels = np.concatenate([
        pd.read_csv(run / s / "subclusters.csv", header=None, dtype=int).values.flatten()[:l]
        for s, l in zip(sessions, lengths)
    ])

    # SUBTLE embeddings (per session, already in shared UMAP space)
    subtle_emb = np.concatenate([
        pd.read_csv(run / s / "embeddings.csv", header=None, dtype=np.float32).values[:l]
        for s, l in zip(sessions, lengths)
    ])
    assert subtle_labels.shape[0] == kp_all.shape[0]

    # -------- 3. B-SOiD (per the original paper, fit on pooled data) --------
    print("\n>>> fitting B-SOiD ...")
    t0 = time.time()
    bsoid = BSOiD(fps=FPS, n_neighbors=60, min_cluster_size=200, umap_dim=3)
    bsoid_res = bsoid.fit(kp_all)
    bsoid_labels = bsoid_res["labels"]               # length T-1 (B-SOiD drops 1 frame)
    bsoid_emb = bsoid_res["embedding_2d"]
    if len(bsoid_labels) < len(gt_all):
        bsoid_labels = np.pad(bsoid_labels, (0, len(gt_all) - len(bsoid_labels)), mode="edge")
        bsoid_emb_pad = np.pad(bsoid_emb, ((0, len(gt_all) - len(bsoid_emb)), (0, 0)),
                               mode="edge")
    else:
        bsoid_emb_pad = bsoid_emb[:len(gt_all)]
    print(f"  B-SOiD: K={bsoid_res['n_clusters']}  ({time.time()-t0:.1f}s)")

    # -------- 4. PCA-HMM (kp-MoSeq fallback) --------
    print("\n>>> fitting PCA-HMM (kp-MoSeq fallback) ...")
    t0 = time.time()
    pca_hmm = _PCAHMMFallback(n_components=8, n_states=25, n_iter=30)
    pca_hmm_res = pca_hmm.fit(kp_all)
    pca_labels = pca_hmm_res.labels
    pca_emb = umap_2d(pca_hmm_res.features, seed=1)
    print(f"  PCA-HMM: K={pca_hmm_res.n_clusters}  ({time.time()-t0:.1f}s)")

    # -------- 5. Metrics --------
    rows = []
    for name, pred, emb in [
        ("SUBTLE",   subtle_labels, subtle_emb),
        ("B-SOiD",   bsoid_labels,  bsoid_emb_pad),
        ("PCA-HMM",  pca_labels,    pca_emb),
    ]:
        # B-SOiD assigns -1 to noise; exclude from purity/ARI for fairness
        if (pred == -1).any():
            mask = pred != -1
            m = metrics(pred[mask], gt_all[mask])
            m["noise_frac"] = float((pred == -1).mean())
        else:
            m = metrics(pred, gt_all)
            m["noise_frac"] = 0.0
        m["method"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("method")[
        ["k", "purity", "ari", "nmi", "noise_frac"]]
    print("\n=== comparison ===")
    print(df.round(3))
    df.to_csv(run / "compare_metrics.csv")

    # -------- 6. Side-by-side UMAP, coloured by predicted cluster --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    for ax, (name, emb, lab) in zip(axes, [
        ("SUBTLE",  subtle_emb,  subtle_labels),
        ("B-SOiD",  bsoid_emb_pad, bsoid_labels),
        ("PCA-HMM", pca_emb,     pca_labels),
    ]):
        # subsample for speed
        n = min(20000, len(lab))
        idx = np.random.default_rng(0).choice(len(lab), n, replace=False)
        sns.scatterplot(x=emb[idx, 0], y=emb[idx, 1], hue=lab[idx],
                        palette="tab20", s=2, ax=ax, legend=False)
        ax.set_title(f"{name}  (K={len(np.unique(lab))})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("UMAP map per method, coloured by predicted cluster (20K subsample)")
    fig.tight_layout(); fig.savefig(run / "compare_umap_by_method.png",
                                    bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {run / 'compare_umap_by_method.png'}")

    # -------- 7. Same UMAP coloured by CalMS21 GT (sanity reference) --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    n = min(20000, len(gt_all))
    idx = np.random.default_rng(0).choice(len(gt_all), n, replace=False)
    gt_sub = gt_all[idx]
    palette = {0: "#d62728", 1: "#1f77b4", 2: "#2ca02c", 3: "#bbbbbb"}
    for ax, (name, emb) in zip(axes, [
        ("SUBTLE",  subtle_emb),
        ("B-SOiD",  bsoid_emb_pad),
        ("PCA-HMM", pca_emb),
    ]):
        for cls in range(4):
            m = gt_sub == cls
            ax.scatter(emb[idx][m, 0], emb[idx][m, 1], s=2,
                       c=palette[cls], label=CLASS_NAMES[cls],
                       alpha=0.5 if cls == 3 else 0.9)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    axes[-1].legend(loc="upper right", framealpha=0.9, markerscale=3)
    fig.suptitle("Same UMAP maps, now coloured by CalMS21 GT label")
    fig.tight_layout(); fig.savefig(run / "compare_umap_by_gt.png",
                                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {run / 'compare_umap_by_gt.png'}")

    # -------- 8. Metric bar chart --------
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    metric_df = df[["purity", "ari", "nmi"]].copy()
    metric_df.plot.bar(ax=ax, colormap="viridis")
    ax.axhline(0.25, ls="--", color="grey", lw=0.8, label="purity chance")
    ax.set_ylabel("score"); ax.set_xlabel("")
    ax.set_title("CalMS21 ground-truth alignment per method")
    ax.set_ylim(0, 1); ax.legend(loc="upper right", ncol=2, fontsize=8)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    fig.tight_layout(); fig.savefig(run / "compare_metrics.png",
                                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {run / 'compare_metrics.png'}")
    print(f"saved: {run / 'compare_metrics.csv'}")


if __name__ == "__main__":
    main()
