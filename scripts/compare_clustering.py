#!/usr/bin/env python3
"""Compare all available clustering/discovery models across datasets.

Models tested:
  1. Clustering (PCA+UMAP+KMeans) — baseline
  2. B-SOiD (UMAP+HDBSCAN+RF) — 2D only
  3. MoSeq fallback (PCA+HMM) — temporal syllables
  4. SUBTLE (Morlet+UMAP+Phenograph) — 3D datasets
  5. hBehaveMAE (hierarchical MAE) — MABe22 pretrained
  6. CEBRA (temporal contrastive learning) — all datasets

Datasets:
  - CalMS21: (N, 2, 64, 7, 2) — 2D mouse pair
  - SUBTLE: (T, 9, 3) — 3D fly
  - Shank3KO: (T, 16, 3) — 3D mouse
  - MABe22: (784, 1800, 36, 2) — 2D multi-mouse

Outputs:
  - Per-model UMAP embedding plots (cluster vs GT)
  - Ethograms
  - Transition matrices
  - Summary metrics table
  - Combined comparison figure

Usage:
    LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 python scripts/compare_clustering.py
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "outputs" / "clustering_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
CMAP = "tab20"


# =====================================================================
# Data loading
# =====================================================================

def load_calms21(n_sequences: int = 2000):
    """Load CalMS21 → (T_total, 7, 2), per-sequence labels."""
    data = np.load(ROOT / "data" / "calms21" / "calms21_aligned.npz")
    x = data["x_train"]  # (19144, 2, 64, 7, 2)
    y = data["y_train"]  # (19144, 4) one-hot

    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), min(n_sequences, len(x)), replace=False)
    kp = x[idx, 0]  # mouse 0: (N, 64, 7, 2)
    labels = np.argmax(y[idx], axis=1)  # (N,)

    # For frame-level models: concatenate sequences
    kp_flat = kp.reshape(-1, 7, 2)  # (N*64, 7, 2)
    labels_flat = np.repeat(labels, 64)

    return {
        "name": "CalMS21",
        "class_names": ["attack", "investigation", "mount", "other"],
        "keypoints": kp_flat,
        "labels": labels_flat,
        "keypoints_seq": kp,       # (N, 64, 7, 2)
        "labels_seq": labels,       # (N,)
        "fps": 30,
        "ndim": 2,
    }


def load_subtle():
    """Load SUBTLE → (T, 9, 3)."""
    csv_dir = ROOT / "data" / "raw" / "subtle"
    csvs = sorted(csv_dir.glob("y5a5_adult_*.csv"))
    if not csvs:
        return None

    # Load first file for demo
    raw = np.loadtxt(csvs[0], delimiter=",")  # (T, 27) = 9 joints × 3D
    T = raw.shape[0]
    kp = raw.reshape(T, 9, 3)

    return {
        "name": "SUBTLE",
        "class_names": None,  # unsupervised — no GT
        "keypoints": kp,
        "labels": None,
        "fps": 20,
        "ndim": 3,
    }


def load_shank3ko():
    """Load Shank3KO → (T, 16, 3) from structured .mat with CoordX/Y/Z fields."""
    mat_path = ROOT / "data" / "raw" / "shank3ko" / "Shank3KO_mice_slk3D.mat"
    if not mat_path.exists():
        return None

    import scipy.io
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
    mice_data = mat.get("mice_slk3D")
    if mice_data is None:
        return None

    # Concat first 5 recordings (cap total frames)
    all_kp = []
    all_genotypes = []
    for i in range(min(5, mice_data.size)):
        rec = mice_data.flat[i]
        cx = rec["CoordX"]  # (T, 16)
        cy = rec["CoordY"]
        cz = rec["CoordZ"]
        coords = np.stack([cx, cy, cz], axis=2)  # (T, 16, 3)
        T = min(coords.shape[0], 10000)  # cap per recording
        all_kp.append(coords[:T].astype(np.float32))
        geno = str(rec["Genotypes"])
        all_genotypes.extend([0 if geno == "KO" else 1] * T)

    kp = np.concatenate(all_kp, axis=0)  # (T_total, 16, 3)
    labels = np.array(all_genotypes[:len(kp)], dtype=int)

    return {
        "name": "Shank3KO",
        "class_names": ["KO", "WT"],
        "keypoints": kp,
        "labels": labels,
        "fps": 30,
        "ndim": 3,
    }


def load_mabe22():
    """Load MABe22 → list of (T, 3, 12, 2) sequences for BehaveMAE."""
    npy_path = ROOT / "data" / "raw" / "mabe22" / "mouse_user_train.npy"
    if not npy_path.exists():
        return None

    raw = np.load(str(npy_path), allow_pickle=True).item()
    sequences = raw.get("sequences", raw)

    if not isinstance(sequences, dict):
        return None

    all_kp = []
    all_annots = []
    for seq_id, seq_data in list(sequences.items())[:100]:  # cap at 100 seqs
        kp = seq_data.get("keypoints")
        if isinstance(kp, np.ndarray) and kp.ndim == 4:
            # (T, 3, 12, 2)
            all_kp.append(kp)
            annot = seq_data.get("annotations")
            if annot is not None:
                all_annots.append(annot)

    if not all_kp:
        return None

    # Concat sequences for frame-level models: (T_total, 3, 12, 2)
    kp_concat = np.concatenate(all_kp, axis=0)
    # Also flatten for generic models: (T_total, 36, 2)
    T = kp_concat.shape[0]
    kp_flat = kp_concat.reshape(T, -1, 2)  # (T, 36, 2)

    return {
        "name": "MABe22",
        "class_names": None,
        "keypoints": kp_flat[:20000],           # (T, 36, 2) for generic models
        "keypoints_4d": kp_concat[:20000],      # (T, 3, 12, 2) for BehaveMAE
        "sequences_4d": all_kp,                 # list of (T_i, 3, 12, 2)
        "labels": None,
        "fps": 30,
        "ndim": 2,
        "n_joints": 36,
    }


# =====================================================================
# Model runners
# =====================================================================

@dataclass
class ModelResult:
    model_name: str
    dataset_name: str
    labels: np.ndarray
    embedding_2d: np.ndarray | None = None
    n_clusters: int = 0
    features: np.ndarray | None = None
    elapsed_sec: float = 0.0
    error: str | None = None
    metadata: dict = field(default_factory=dict)


def run_clustering(dataset: dict, n_clusters: int = 8) -> ModelResult:
    """PCA+UMAP+KMeans baseline."""
    from behavior_lab.data.features import SkeletonBackend
    from behavior_lab.models.discovery.clustering import cluster_features

    t0 = time.time()
    backend = SkeletonBackend(fps=dataset["fps"])
    features = backend.extract(dataset["keypoints"])

    result = cluster_features(features, n_clusters=n_clusters, use_umap=True)
    elapsed = time.time() - t0

    return ModelResult(
        model_name="Clustering",
        dataset_name=dataset["name"],
        labels=result["labels"],
        embedding_2d=result["embedding_2d"],
        n_clusters=result["n_clusters"],
        features=features,
        elapsed_sec=elapsed,
    )


def run_bsoid(dataset: dict) -> ModelResult:
    """B-SOiD: UMAP+HDBSCAN."""
    if dataset["ndim"] != 2:
        return ModelResult("B-SOiD", dataset["name"], np.array([]),
                           error="2D only")

    from behavior_lab.models.discovery.bsoid import BSOiD

    t0 = time.time()
    model = BSOiD(fps=dataset["fps"], min_cluster_size=50)
    result = model.fit(dataset["keypoints"])
    elapsed = time.time() - t0

    return ModelResult(
        model_name="B-SOiD",
        dataset_name=dataset["name"],
        labels=result["labels"],
        embedding_2d=result["embedding_2d"],
        n_clusters=result["n_clusters"],
        features=result.get("features"),
        elapsed_sec=elapsed,
    )


def run_moseq_fallback(dataset: dict, n_states: int = 15) -> ModelResult:
    """PCA+HMM fallback for MoSeq."""
    from behavior_lab.models.discovery.moseq import _PCAHMMFallback

    t0 = time.time()
    # Cap frames for speed
    kp = dataset["keypoints"][:20000]
    model = _PCAHMMFallback(n_components=10, n_states=n_states, n_iter=100)
    cr = model.fit(kp)
    elapsed = time.time() - t0

    return ModelResult(
        model_name="MoSeq (HMM)",
        dataset_name=dataset["name"],
        labels=cr.labels,
        embedding_2d=cr.embeddings,
        n_clusters=cr.n_clusters,
        features=cr.features,
        elapsed_sec=elapsed,
        metadata=cr.metadata,
    )


def run_subtle(dataset: dict) -> ModelResult:
    """SUBTLE: Morlet wavelet + UMAP + Phenograph.

    Runs in a subprocess to isolate potential SIGSEGV from macOS
    multiprocessing issues (loky/OMP pthread_mutex_init).
    """
    import json
    import subprocess
    import tempfile

    kp = dataset["keypoints"][:20000]  # cap

    # Save keypoints to temp file for subprocess
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
        np.savez(f, keypoints=kp)

    out_path = tmp_path.replace(".npz", "_result.npz")

    # Run SUBTLE in isolated subprocess
    script = f"""
import sys, json, time
sys.path.insert(0, 'src')
import numpy as np
from behavior_lab.models.discovery.subtle_wrapper import SUBTLE

kp = np.load('{tmp_path}')['keypoints']
t0 = time.time()
model = SUBTLE(fps={dataset['fps']})
cr = model.fit_predict([kp])
elapsed = time.time() - t0

np.savez('{out_path}',
    labels=cr.labels,
    embeddings=cr.embeddings if cr.embeddings is not None else np.array([]),
    n_clusters=np.array(cr.n_clusters),
    elapsed=np.array(elapsed),
)
print(json.dumps({{'n_clusters': int(cr.n_clusters), 'elapsed': elapsed}}))
"""

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=600,
            env={**__import__("os").environ,
                 "LOKY_MAX_CPU_COUNT": "1", "OMP_NUM_THREADS": "1"},
        )
        if proc.returncode != 0:
            stderr_short = proc.stderr[-500:] if proc.stderr else "unknown"
            return ModelResult("SUBTLE", dataset["name"], np.array([]),
                               error=f"Subprocess exit {proc.returncode}: {stderr_short}")

        data = np.load(out_path)
        elapsed = float(data["elapsed"])
        labels = data["labels"]
        emb = data["embeddings"]
        if emb.size == 0:
            emb = None

        return ModelResult(
            model_name="SUBTLE",
            dataset_name=dataset["name"],
            labels=labels,
            embedding_2d=emb,
            n_clusters=int(data["n_clusters"]),
            elapsed_sec=elapsed,
        )

    except subprocess.TimeoutExpired:
        return ModelResult("SUBTLE", dataset["name"], np.array([]),
                           error="Timeout (600s)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def run_behavemae(dataset: dict) -> ModelResult:
    """hBehaveMAE: hierarchical MAE encoder → clustering.

    Uses sliding window (900 frames) to extract per-window embeddings,
    then clusters the pooled feature vectors.
    """
    ckpt = ROOT / "checkpoints" / "behavemae" / "hBehaveMAE_MABe22.pth"
    if not ckpt.exists():
        return ModelResult("hBehaveMAE", dataset["name"], np.array([]),
                           error="No checkpoint")

    # BehaveMAE needs (T, 3, 12, 2) MABe22 format
    kp_4d = dataset.get("keypoints_4d")
    seqs_4d = dataset.get("sequences_4d")
    if kp_4d is None:
        return ModelResult("hBehaveMAE", dataset["name"], np.array([]),
                           error="Need MABe22 4D keypoints")

    from behavior_lab.models.discovery.behavemae import BehaveMAE
    from behavior_lab.models.discovery.clustering import cluster_features

    t0 = time.time()
    model = BehaveMAE.from_pretrained(
        checkpoint_path=str(ckpt), dataset="mabe22", device="cpu"
    )

    # Extract per-window features via sliding window
    window = 900
    stride = 450
    all_features = []

    # Use individual sequences if available, else slide over concat
    sources = seqs_4d if seqs_4d else [kp_4d]
    for seq in sources[:50]:  # cap sequences
        T = seq.shape[0]
        for start in range(0, max(T - window + 1, 1), stride):
            chunk = seq[start:start + window]
            emb = model.encode(chunk)  # (N_tokens, 256)
            all_features.append(emb.mean(axis=0))  # pool → (256,)

    if not all_features:
        return ModelResult("hBehaveMAE", dataset["name"], np.array([]),
                           error="No features extracted")

    features = np.stack(all_features)  # (N_windows, 256)
    print(f"      BehaveMAE: {features.shape[0]} windows × {features.shape[1]}D")

    result = cluster_features(features, n_clusters=8,
                              use_umap=features.shape[0] > 15)
    elapsed = time.time() - t0

    return ModelResult(
        model_name="hBehaveMAE",
        dataset_name=dataset["name"],
        labels=result["labels"],
        embedding_2d=result["embedding_2d"],
        n_clusters=result["n_clusters"],
        features=features,
        elapsed_sec=elapsed,
    )


def run_cebra(dataset: dict, output_dim: int = 32, n_clusters: int = 8,
              max_iterations: int = 5000) -> ModelResult:
    """CEBRA: temporal contrastive learning → clustering.

    Fits CEBRA on raw keypoint time-series (preserving temporal order),
    then clusters the learned embeddings.
    """
    from behavior_lab.data.features.cebra_backend import CEBRABackend
    from behavior_lab.models.discovery.clustering import cluster_features

    t0 = time.time()
    kp = dataset["keypoints"][:20000]  # cap frames

    backend = CEBRABackend(
        output_dim=output_dim,
        max_iterations=max_iterations,
        time_offsets=10,
        device="cpu",
    )
    embeddings = backend.extract(kp)  # (T, output_dim)
    print(f"      CEBRA: {kp.shape} → {embeddings.shape}")

    result = cluster_features(embeddings, n_clusters=n_clusters, use_umap=True)
    elapsed = time.time() - t0

    return ModelResult(
        model_name="CEBRA",
        dataset_name=dataset["name"],
        labels=result["labels"],
        embedding_2d=result["embedding_2d"],
        n_clusters=result["n_clusters"],
        features=embeddings,
        elapsed_sec=elapsed,
    )


# =====================================================================
# Visualization
# =====================================================================

def compute_metrics(labels: np.ndarray, features: np.ndarray | None,
                    gt_labels: np.ndarray | None) -> dict:
    """Compute clustering quality metrics."""
    from sklearn.metrics import silhouette_score

    metrics = {"n_clusters": len(set(labels) - {-1})}

    if features is not None and len(set(labels) - {-1}) > 1:
        valid = labels >= 0
        if valid.sum() > 10 and len(features) == len(labels):
            try:
                metrics["silhouette"] = float(
                    silhouette_score(features[valid], labels[valid],
                                     sample_size=min(5000, valid.sum())))
            except Exception:
                pass

    if gt_labels is not None and len(gt_labels) == len(labels):
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        valid = labels >= 0
        if valid.sum() > 0:
            gt_valid = gt_labels[:len(labels)][valid]
            metrics["ARI"] = float(adjusted_rand_score(gt_valid, labels[valid]))
            metrics["NMI"] = float(normalized_mutual_info_score(gt_valid, labels[valid]))

    return metrics


def plot_model_result(result: ModelResult, gt_labels: np.ndarray | None,
                      class_names: list[str] | None, row_axes):
    """Plot one model result on a row of axes: [embedding, ethogram, transition]."""
    import matplotlib.pyplot as plt

    ax_emb, ax_eth, ax_trans = row_axes

    if result.error:
        for ax in row_axes:
            ax.text(0.5, 0.5, f"{result.model_name}\n{result.error}",
                    ha="center", va="center", fontsize=10, color="gray",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return

    labels = result.labels
    n_clusters = result.n_clusters

    # Embedding
    if result.embedding_2d is not None and result.embedding_2d.shape[0] > 1:
        emb = result.embedding_2d
        cmap_obj = plt.colormaps.get_cmap(CMAP)
        if gt_labels is not None and len(gt_labels) == len(labels):
            c = gt_labels[:len(emb)]
            ax_emb.scatter(emb[:, 0], emb[:, 1], c=c, cmap=CMAP,
                          alpha=0.3, s=1, rasterized=True)
            ax_emb.set_title(f"{result.model_name} (GT colored)", fontsize=9)
        else:
            ax_emb.scatter(emb[:, 0], emb[:, 1], c=labels[:len(emb)], cmap=CMAP,
                          alpha=0.3, s=1, rasterized=True)
            ax_emb.set_title(f"{result.model_name} ({n_clusters} clusters)", fontsize=9)
        ax_emb.set_xticks([])
        ax_emb.set_yticks([])
    else:
        ax_emb.text(0.5, 0.5, "No embedding", ha="center", va="center",
                    transform=ax_emb.transAxes, fontsize=9, color="gray")

    # Ethogram (first 3000 frames)
    n_show = min(3000, len(labels))
    seg = labels[:n_show]
    unique = sorted(set(seg))
    if len(unique) > 0:
        from matplotlib.colors import ListedColormap
        n_colors = max(len(unique), 2)
        base_cmap = plt.colormaps[CMAP]
        color_cmap = ListedColormap([base_cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)])
        label_map = {l: i for i, l in enumerate(unique)}
        colors = np.array([label_map[l] for l in seg])
        ax_eth.imshow(colors.reshape(1, -1), aspect="auto", cmap=color_cmap,
                      interpolation="nearest")
        ax_eth.set_yticks([])
        ax_eth.set_title(f"Ethogram ({n_show} frames)", fontsize=9)
    else:
        ax_eth.text(0.5, 0.5, "No labels", ha="center", va="center",
                    transform=ax_eth.transAxes)

    # Transition matrix
    if n_clusters > 1 and n_clusters <= 30:
        trans = np.zeros((n_clusters, n_clusters))
        valid_labels = labels[labels >= 0]
        for i in range(len(valid_labels) - 1):
            a, b = valid_labels[i], valid_labels[i + 1]
            if a < n_clusters and b < n_clusters:
                trans[a, b] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans = trans / row_sums
        ax_trans.imshow(trans, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
        ax_trans.set_title(f"Transitions ({n_clusters}×{n_clusters})", fontsize=9)
        ax_trans.set_xlabel("To", fontsize=7)
        ax_trans.set_ylabel("From", fontsize=7)
        ax_trans.tick_params(labelsize=6)
    else:
        ax_trans.text(0.5, 0.5, f"{n_clusters} clusters", ha="center",
                      va="center", transform=ax_trans.transAxes, fontsize=9)


def make_comparison_figure(results: list[ModelResult], dataset: dict):
    """Create multi-row comparison figure for one dataset."""
    import matplotlib.pyplot as plt

    valid_results = [r for r in results if r.labels is not None and len(r.labels) > 0]
    if not valid_results:
        # Include error results for display
        valid_results = results

    n_models = len(valid_results)
    fig, axes = plt.subplots(n_models, 3, figsize=(16, 3.5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    gt_labels = dataset.get("labels")
    class_names = dataset.get("class_names")

    for i, result in enumerate(valid_results):
        plot_model_result(result, gt_labels, class_names, axes[i])

    fig.suptitle(f"Clustering Comparison — {dataset['name']}", fontsize=14, y=1.01)
    fig.tight_layout()

    save_path = OUTPUT_DIR / f"comparison_{dataset['name'].lower()}.png"
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def print_metrics_table(all_results: list[ModelResult], all_datasets: list[dict]):
    """Print summary metrics table."""
    print("\n" + "=" * 80)
    print(f"{'Model':<16} {'Dataset':<12} {'Clusters':>8} {'Silh':>8} {'ARI':>8} {'NMI':>8} {'Time':>8}")
    print("-" * 80)

    for ds in all_datasets:
        gt = ds.get("labels")
        for r in all_results:
            if r.dataset_name != ds["name"]:
                continue
            if r.error:
                print(f"{r.model_name:<16} {r.dataset_name:<12} {'ERR':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8}  ({r.error})")
                continue

            m = compute_metrics(r.labels, r.features, gt)
            sil = f"{m.get('silhouette', float('nan')):.3f}" if 'silhouette' in m else "—"
            ari = f"{m.get('ARI', float('nan')):.3f}" if 'ARI' in m else "—"
            nmi = f"{m.get('NMI', float('nan')):.3f}" if 'NMI' in m else "—"
            print(f"{r.model_name:<16} {r.dataset_name:<12} {m['n_clusters']:>8} {sil:>8} {ari:>8} {nmi:>8} {r.elapsed_sec:>7.1f}s")
    print("=" * 80)


# =====================================================================
# Main
# =====================================================================

def main():
    print("Loading datasets...")
    datasets = []

    # CalMS21
    ds = load_calms21(n_sequences=2000)
    datasets.append(ds)
    print(f"  CalMS21: {ds['keypoints'].shape}, labels: {ds['labels'].shape}")

    # SUBTLE
    ds = load_subtle()
    if ds:
        datasets.append(ds)
        print(f"  SUBTLE: {ds['keypoints'].shape}")

    # Shank3KO
    ds = load_shank3ko()
    if ds:
        datasets.append(ds)
        print(f"  Shank3KO: {ds['keypoints'].shape}")

    # MABe22
    ds = load_mabe22()
    if ds:
        datasets.append(ds)
        print(f"  MABe22: {ds['keypoints'].shape} (4D: {ds['keypoints_4d'].shape})")

    all_results: list[ModelResult] = []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset['name']} ({dataset['keypoints'].shape})")
        print(f"{'='*60}")

        ds_results = []

        # 1. Clustering baseline
        print("  [1] Clustering (PCA+UMAP+KMeans)...")
        try:
            r = run_clustering(dataset, n_clusters=8)
            ds_results.append(r)
            print(f"      → {r.n_clusters} clusters, {r.elapsed_sec:.1f}s")
        except Exception as e:
            print(f"      → FAILED: {e}")
            ds_results.append(ModelResult("Clustering", dataset["name"], np.array([]), error=str(e)))

        # 2. B-SOiD (2D only)
        print("  [2] B-SOiD (UMAP+HDBSCAN)...")
        try:
            r = run_bsoid(dataset)
            ds_results.append(r)
            if r.error:
                print(f"      → SKIP: {r.error}")
            else:
                print(f"      → {r.n_clusters} clusters, {r.elapsed_sec:.1f}s")
        except Exception as e:
            print(f"      → FAILED: {e}")
            ds_results.append(ModelResult("B-SOiD", dataset["name"], np.array([]), error=str(e)))

        # 3. MoSeq HMM fallback
        print("  [3] MoSeq (PCA+HMM)...")
        try:
            r = run_moseq_fallback(dataset, n_states=15)
            ds_results.append(r)
            print(f"      → {r.n_clusters} clusters, {r.elapsed_sec:.1f}s")
        except Exception as e:
            print(f"      → FAILED: {e}")
            traceback.print_exc()
            ds_results.append(ModelResult("MoSeq (HMM)", dataset["name"], np.array([]), error=str(e)))

        # 4. SUBTLE (3D preferred)
        print("  [4] SUBTLE (Morlet+Phenograph)...")
        try:
            r = run_subtle(dataset)
            ds_results.append(r)
            print(f"      → {r.n_clusters} clusters, {r.elapsed_sec:.1f}s")
        except Exception as e:
            print(f"      → FAILED: {e}")
            ds_results.append(ModelResult("SUBTLE", dataset["name"], np.array([]), error=str(e)))

        # 5. BehaveMAE (MABe22 only)
        if dataset.get("keypoints_4d") is not None:
            print("  [5] hBehaveMAE (hierarchical MAE)...")
            try:
                r = run_behavemae(dataset)
                ds_results.append(r)
                if r.error:
                    print(f"      → SKIP: {r.error}")
                else:
                    print(f"      → {r.n_clusters} clusters, {r.elapsed_sec:.1f}s")
            except Exception as e:
                print(f"      → FAILED: {e}")
                traceback.print_exc()
                ds_results.append(ModelResult("hBehaveMAE", dataset["name"],
                                              np.array([]), error=str(e)))

        # 6. CEBRA (temporal contrastive learning)
        print("  [6] CEBRA (temporal contrastive)...")
        try:
            r = run_cebra(dataset, output_dim=32, n_clusters=8, max_iterations=2000)
            ds_results.append(r)
            print(f"      → {r.n_clusters} clusters, {r.elapsed_sec:.1f}s")
        except Exception as e:
            print(f"      → FAILED: {e}")
            traceback.print_exc()
            ds_results.append(ModelResult("CEBRA", dataset["name"],
                                          np.array([]), error=str(e)))

        all_results.extend(ds_results)

        # Generate comparison figure
        print("  Generating comparison figure...")
        make_comparison_figure(ds_results, dataset)

    # Summary table
    print_metrics_table(all_results, datasets)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
