#!/usr/bin/env python3
"""Run a capped, notebook-friendly behavior discovery comparison.

This runner executes all installed discovery families on small, consistent
local dataset slices (self-contained convenience CLI).

For the canonical, config-driven cross-species comparison + report, prefer the
library entry point ``behavior_lab.experiments.run_comparison`` (ties ingest ->
compare_discovery_methods -> render_comparison_report, with ground-truth ARI/NMI
and per-cluster galleries). Heavy methods run in isolated envs via
``scripts/isolated_run.py``.
"""
# no-split: cohesive comparison-runner script — run_* fns share DatasetSlice/
# BatchResult/metric_result + plot/html helpers; splitting fragments locality.
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavior_lab.data import ingest
from behavior_lab.data.features import SkeletonBackend
from behavior_lab.data.preprocessing import Interpolator
from behavior_lab.evaluation import compute_behavior_metrics
from behavior_lab.models.discovery.bsoid import BSOiD
from behavior_lab.models.discovery.clustering import cluster_features
from behavior_lab.models.discovery.moseq import _PCAHMMFallback


OUT_DIR = ROOT / "outputs" / "behavior_analysis_workbench" / "batch"
RANDOM_STATE = 42
MAX_FRAMES = 1500


@dataclass
class DatasetSlice:
    name: str
    keypoints: np.ndarray
    fps: float
    labels: np.ndarray | None = None
    keypoints_4d: np.ndarray | None = None
    notes: dict[str, object] = field(default_factory=dict)


@dataclass
class BatchResult:
    dataset: str
    method: str
    status: str
    n_frames: int
    n_clusters: int | None = None
    elapsed_sec: float | None = None
    silhouette: float | None = None
    noise_frac: float | None = None
    ari: float | None = None
    nmi: float | None = None
    num_bouts: int | None = None
    mean_bout_sec: float | None = None
    labels_path: str | None = None
    embedding_path: str | None = None
    error: str | None = None
    notes: dict[str, object] = field(default_factory=dict)


def load_datasets(max_frames: int = MAX_FRAMES) -> list[DatasetSlice]:
    datasets: list[DatasetSlice] = []

    calms = ROOT / "data" / "calms21" / "calms21_aligned.npz"
    if calms.exists():
        d = np.load(calms, allow_pickle=True)
        x = d["x_train"]
        y = d["y_train"].argmax(axis=1)
        n_seq = min(max(1, math.ceil(max_frames / x.shape[2])), len(x))
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(x), n_seq, replace=False)
        kp = x[idx].transpose(0, 2, 1, 3, 4).reshape(-1, 14, 2)[:max_frames]
        labels = np.repeat(y[idx], x.shape[2])[: len(kp)]
        datasets.append(DatasetSlice(
            name="calms21",
            keypoints=kp.astype(np.float32),
            labels=labels.astype(int),
            fps=30.0,
            notes={"source": str(calms), "shape": str(kp.shape), "sampled_sequences": int(n_seq)},
        ))

    subtle = ROOT / "data" / "preprocessed" / "subtle" / "subtle_all.npz"
    if subtle.exists():
        kp = np.load(subtle, allow_pickle=True)["keypoints"][:max_frames].astype(np.float32)
        datasets.append(DatasetSlice(
            name="subtle",
            keypoints=kp,
            fps=20.0,
            notes={"source": str(subtle), "shape": str(kp.shape)},
        ))

    shank_dir = ROOT / "data" / "preprocessed" / "shank3ko"
    if shank_dir.exists():
        parts = []
        labels = []
        for label, pattern in [(0, "*_KO_*.npz"), (1, "*_WT_*.npz")]:
            paths = sorted(shank_dir.glob(pattern))
            if not paths:
                continue
            kp = np.load(paths[0], allow_pickle=True)["keypoints"][: max_frames // 2].astype(np.float32)
            parts.append(kp)
            labels.append(np.full(len(kp), label, dtype=int))
        if parts:
            kp = np.concatenate(parts, axis=0)[:max_frames]
            lab = np.concatenate(labels, axis=0)[: len(kp)]
            datasets.append(DatasetSlice(
                name="shank3ko",
                keypoints=kp,
                labels=lab,
                fps=30.0,
                notes={"source": str(shank_dir), "shape": str(kp.shape), "labels": "0=KO,1=WT"},
            ))

    mabe = ROOT / "data" / "preprocessed" / "mabe22" / "mouse_user_train.npz"
    if mabe.exists():
        d = np.load(mabe, allow_pickle=True)
        kp_seq = d["keypoints"][:2].astype(np.float32)  # (N,T,36,2)
        kp = kp_seq.reshape(-1, 36, 2)[:max_frames]
        kp4 = kp_seq.reshape(kp_seq.shape[0], kp_seq.shape[1], 3, 12, 2)
        labels = None
        if "annotations" in d:
            ann = d["annotations"][:2, 0].reshape(-1)[: len(kp)]
            labels = ann.astype(int)
        datasets.append(DatasetSlice(
            name="mabe22",
            keypoints=kp,
            labels=labels,
            keypoints_4d=kp4,
            fps=30.0,
            notes={"source": str(mabe), "shape": str(kp.shape), "sequences": 2},
        ))

    # AVATAR 3D keypoints (BS lane output). One slice per post-processing variant, native joint
    # layout from the file (11 points), not remapped to SUBTLE's 9 (docs/architecture.md).
    avatar_dir = Path(os.environ.get("BEHAVIOR_LAB_AVATAR_DIR",
                                     "~/data/avatar_gslrm/keypoints")).expanduser()
    paths = sorted(avatar_dir.glob("*_gslrm.npz"))
    prefix = os.path.commonprefix([p.stem for p in paths]) if len(paths) > 1 else ""
    for path in paths:
        seq = ingest(path, units="gslrm_normalized")[0]
        variant = path.stem[len(prefix):].removesuffix("gslrm").strip("_") or "base"
        datasets.append(DatasetSlice(
            name=f"avatar_{variant}",
            keypoints=seq.keypoints[:max_frames],
            fps=20.0,  # not stored in the file; AVATAR lane contract
            notes={"source": str(path), "shape": str(seq.keypoints.shape),
                   "units": "gslrm_normalized", "node_names": seq.metadata.get("node_names")},
        ))

    return [fill_missing(ds) for ds in datasets]


def fill_missing(ds: DatasetSlice) -> DatasetSlice:
    """Interpolate NaN over time and record how much (zero-fill would be a real position)."""
    miss = np.isnan(ds.keypoints)
    ds.notes["nan_frac"] = round(float(miss.mean()), 4)
    if miss.any():
        m = miss.reshape(len(miss), -1).T  # (channels, T)
        runs = np.diff(np.pad(m.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        # np.nonzero is row-major, so starts and ends pair up channel by channel
        gaps = np.nonzero(runs == -1)[1] - np.nonzero(runs == 1)[1]
        assert gaps.sum() == miss.reshape(len(miss), -1).sum()
        ds.notes["max_gap_frames"] = int(gaps.max())
        fill = Interpolator(max_gap=len(ds.keypoints))
        ds.keypoints = fill(ds.keypoints).astype(np.float32)
    return ds


def metric_result(ds: DatasetSlice, method: str, labels: np.ndarray, features: np.ndarray | None,
                  embedding: np.ndarray | None, elapsed: float, notes: dict[str, object] | None = None,
                  label_fps: float | None = None) -> BatchResult:
    valid = labels >= 0
    n_clusters = len(set(labels[valid]) if valid.any() else set())
    sil = None
    if features is not None and len(features) == len(labels) and n_clusters > 1 and valid.sum() > n_clusters:
        sil = float(silhouette_score(features[valid], labels[valid], sample_size=min(1000, valid.sum())))

    ari = nmi = None
    if ds.labels is not None and len(ds.labels) == len(labels):
        ari = float(adjusted_rand_score(ds.labels[valid], labels[valid])) if valid.any() else None
        nmi = float(normalized_mutual_info_score(ds.labels[valid], labels[valid])) if valid.any() else None

    behavior = compute_behavior_metrics(labels, fps=label_fps or ds.fps)
    mean_bout = float(np.mean(list(behavior.bout_durations.values()))) if behavior.bout_durations else None

    ds_dir = OUT_DIR / "arrays" / ds.name
    ds_dir.mkdir(parents=True, exist_ok=True)
    safe = method.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    labels_path = ds_dir / f"{safe}_labels.npy"
    np.save(labels_path, labels)

    embedding_path = None
    if embedding is not None:
        embedding_path = ds_dir / f"{safe}_embedding.npy"
        np.save(embedding_path, embedding)

    return BatchResult(
        dataset=ds.name,
        method=method,
        status="ok",
        n_frames=int(len(labels)),
        n_clusters=int(n_clusters),
        elapsed_sec=float(elapsed),
        silhouette=sil,
        noise_frac=float((~valid).mean()),
        ari=ari,
        nmi=nmi,
        num_bouts=int(behavior.num_bouts),
        mean_bout_sec=mean_bout,
        labels_path=str(labels_path.relative_to(ROOT)),
        embedding_path=str(embedding_path.relative_to(ROOT)) if embedding_path else None,
        notes=notes or {},
    )


def error_result(ds: DatasetSlice, method: str, exc: BaseException | str) -> BatchResult:
    err = str(exc)
    return BatchResult(
        dataset=ds.name,
        method=method,
        status="error",
        n_frames=int(ds.keypoints.shape[0]),
        error=err[-1000:],
    )


def run_kmeans(ds: DatasetSlice) -> BatchResult:
    t0 = time.time()
    features = SkeletonBackend(fps=ds.fps, normalize_body_size=True).extract(ds.keypoints)
    out = cluster_features(features, n_clusters=8, use_umap=True, random_state=RANDOM_STATE)
    return metric_result(ds, "kmeans_pca_umap", out["labels"], features, out["embedding_2d"], time.time() - t0)


def run_bsoid(ds: DatasetSlice) -> BatchResult:
    t0 = time.time()
    out = BSOiD(fps=int(ds.fps), min_cluster_size=20, random_state=RANDOM_STATE).fit(ds.keypoints)
    return metric_result(
        ds, "B-SOiD", out["labels"], out.get("features"), out.get("embedding_2d"),
        time.time() - t0, notes={"label_rate": "10fps bins"},
        label_fps=ds.fps / max(1, int(ds.fps) // 10),  # labels are per bin, not per frame
    )


def run_pca_hmm(ds: DatasetSlice) -> BatchResult:
    t0 = time.time()
    cr = _PCAHMMFallback(n_components=10, n_states=12, n_iter=50).fit(ds.keypoints)
    return metric_result(ds, "pca_hmm_moseq_fallback", cr.labels, cr.features, cr.embeddings, time.time() - t0)


def run_keypoint_moseq(ds: DatasetSlice) -> BatchResult:
    from behavior_lab.models.discovery.moseq import KeypointMoSeq

    t0 = time.time()
    kp = ds.keypoints[: min(300, len(ds.keypoints))]
    model = KeypointMoSeq(
        project_dir=str(OUT_DIR / "keypoint_moseq" / ds.name),
        num_iters=5,
        latent_dim=min(6, max(2, kp.shape[1])),
        bodypart_names=[f"kp{i}" for i in range(kp.shape[1])],
    )
    cr = model.fit_predict(kp)
    small_ds = DatasetSlice(ds.name, kp, ds.fps, ds.labels[: len(kp)] if ds.labels is not None else None)
    return metric_result(small_ds, "keypoint_moseq", cr.labels, cr.features, cr.embeddings, time.time() - t0,
                         notes={"max_frames": len(kp), "num_iters": 5})


def run_subtle(ds: DatasetSlice) -> BatchResult:
    if ds.keypoints.shape[-1] != 3:
        raise ValueError("SUBTLE route is run only on 3D slices in this batch")
    from behavior_lab.models.discovery.subtle_wrapper import SUBTLE

    t0 = time.time()
    cr = SUBTLE(fps=int(ds.fps)).fit_predict(
        [ds.keypoints[: min(1200, len(ds.keypoints))]],
        isolate=True,
    )
    small_ds = DatasetSlice(ds.name, ds.keypoints[: len(cr.labels)], ds.fps,
                            ds.labels[: len(cr.labels)] if ds.labels is not None else None)
    return metric_result(small_ds, "SUBTLE", cr.labels, cr.features, cr.embeddings, time.time() - t0,
                         notes={"max_frames": len(cr.labels)})


def run_behavemae(ds: DatasetSlice) -> BatchResult:
    if ds.keypoints_4d is None:
        raise ValueError("hBehaveMAE route requires MABe22 4D keypoints")
    ckpt = ROOT / "checkpoints" / "behavemae" / "hBehaveMAE_MABe22.pth"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    from behavior_lab.models.discovery.behavemae import BehaveMAE

    t0 = time.time()
    model = BehaveMAE.from_pretrained(checkpoint_path=str(ckpt), dataset="mabe22", device="cpu")
    features = []
    for seq in ds.keypoints_4d[:2]:
        chunk = seq[:900]
        if len(chunk) == 900:
            features.append(model.encode(chunk).mean(axis=0))
    if not features:
        raise RuntimeError("No 900-frame MABe22 chunks available")
    features = np.stack(features)
    n_clusters = min(2, len(features))
    out = cluster_features(features, n_clusters=n_clusters, use_umap=False, random_state=RANDOM_STATE)
    pseudo_ds = DatasetSlice(ds.name, ds.keypoints[: len(out["labels"])], ds.fps)
    return metric_result(pseudo_ds, "hBehaveMAE", out["labels"], features, out["embedding_2d"], time.time() - t0,
                         notes={"windows": len(features), "checkpoint": str(ckpt.relative_to(ROOT))})


def run_cebra(ds: DatasetSlice) -> BatchResult:
    from behavior_lab.data.features.cebra_backend import CEBRABackend

    t0 = time.time()
    kp = ds.keypoints[: min(1200, len(ds.keypoints))]
    features = CEBRABackend(output_dim=16, max_iterations=100, time_offsets=5, device="cpu").extract(kp)
    out = cluster_features(features, n_clusters=8, use_umap=True, random_state=RANDOM_STATE)
    small_ds = DatasetSlice(ds.name, kp, ds.fps, ds.labels[: len(kp)] if ds.labels is not None else None)
    return metric_result(small_ds, "CEBRA", out["labels"], features, out["embedding_2d"], time.time() - t0,
                         notes={"max_iterations": 100, "output_dim": 16})


def run_vame(ds: DatasetSlice) -> BatchResult:
    from behavior_lab.models import get_model

    t0 = time.time()
    kp = ds.keypoints[: min(1200, len(ds.keypoints))]
    model = get_model(
        "vame",
        project_dir=str(OUT_DIR / "vame" / ds.name),
        n_clusters=8,
        num_epochs=20,
        fps=ds.fps,
    )
    cr = model.fit_predict(kp)
    small_ds = DatasetSlice(ds.name, ds.keypoints[: len(cr.labels)], ds.fps,
                            ds.labels[: len(cr.labels)] if ds.labels is not None else None)
    return metric_result(small_ds, "VAME", cr.labels, cr.features, cr.embeddings, time.time() - t0,
                         notes={"max_frames": len(cr.labels), "num_epochs": 20})


METHODS: dict[str, Callable[[DatasetSlice], BatchResult]] = {
    "kmeans_pca_umap": run_kmeans,
    "B-SOiD": run_bsoid,
    "pca_hmm_moseq_fallback": run_pca_hmm,
    "keypoint_moseq": run_keypoint_moseq,
    "SUBTLE": run_subtle,
    "VAME": run_vame,
    "hBehaveMAE": run_behavemae,
    "CEBRA": run_cebra,
}


def plot_summary(df: pd.DataFrame) -> None:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    pivot = ok.pivot_table(index="dataset", columns="method", values="silhouette", aggfunc="first")
    im = axes[0].imshow(pivot.fillna(np.nan), aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_title("Silhouette by dataset/method")
    fig.colorbar(im, ax=axes[0], label="silhouette")

    for i, dataset in enumerate(pivot.index):
        for j, method in enumerate(pivot.columns):
            val = pivot.loc[dataset, method]
            if pd.notna(val):
                axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)

    time_pivot = ok.pivot_table(index="dataset", columns="method", values="elapsed_sec", aggfunc="first")
    time_pivot.plot(kind="bar", ax=axes[1])
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Runtime by dataset/method")
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(OUT_DIR / "batch_summary.png", dpi=150)
    plt.close(fig)


def write_html(df: pd.DataFrame, slices: list[dict]) -> None:
    rows = df.fillna("").to_dict(orient="records")
    html_rows = "\n".join(
        "<tr>" + "".join(f"<td>{row.get(col, '')}</td>" for col in df.columns) + "</tr>"
        for row in rows
    )
    dataset_rows = "\n".join(
        f"<tr><td>{d['name']}</td><td>{d['shape']}</td><td>{d['fps']}</td><td>{d['has_labels']}</td><td>{d['notes']}</td></tr>"
        for d in slices
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Behavior Workbench Batch</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;line-height:1.4}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:6px;vertical-align:top}}
th{{background:#f4f4f4}}img{{max-width:1100px;width:100%;height:auto}}</style></head>
<body>
<h1>Behavior Analysis Workbench Batch</h1>
<p>Seed={RANDOM_STATE}, max_frames={MAX_FRAMES}. Errors are retained to make missing or incompatible routes explicit.</p>
<h2>Datasets</h2><table><tr><th>Name</th><th>Keypoints</th><th>FPS</th><th>Labels</th><th>Notes</th></tr>{dataset_rows}</table>
<h2>Summary</h2><img src="batch_summary.png">
<h2>Results</h2><table><tr>{''.join(f'<th>{c}</th>' for c in df.columns)}</tr>{html_rows}</table>
</body></html>"""
    (OUT_DIR / "batch_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets",
                    help="comma list, prefix match, e.g. avatar,subtle (default: all found)")
    ap.add_argument("--methods", help=f"comma list from: {', '.join(METHODS)} (default: all)")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(args.max_frames)
    if args.datasets:
        want = args.datasets.split(",")
        datasets = [d for d in datasets if any(d.name.startswith(w) for w in want)]
    methods = {m: METHODS[m] for m in args.methods.split(",")} if args.methods else METHODS
    assert datasets, "no dataset matched"
    results: list[BatchResult] = []

    for ds in datasets:
        print(f"\nDataset {ds.name}: {ds.keypoints.shape}, fps={ds.fps}, "
              f"labels={ds.labels is not None}, nan_frac={ds.notes.get('nan_frac')}")
        for method, fn in methods.items():
            print(f"  {method}...", flush=True)
            try:
                result = fn(ds)
                print(f"    ok: clusters={result.n_clusters}, sil={result.silhouette}, {result.elapsed_sec:.1f}s")
            except Exception as exc:
                traceback.print_exc(limit=2)
                result = error_result(ds, method, exc)
                print(f"    error: {result.error}")
            results.append(result)

    # Upsert by (dataset, method): methods live in separate envs, so one run fills only some cells.
    rows = {(r.dataset, r.method): asdict(r) for r in results}
    prev = OUT_DIR / "batch_results.json"
    if prev.exists():
        rows = {(r["dataset"], r["method"]): r for r in json.loads(prev.read_text())} | rows
    df = pd.DataFrame(list(rows.values()))
    df.to_csv(OUT_DIR / "batch_results.csv", index=False)
    prev.write_text(json.dumps(list(rows.values()), indent=2), encoding="utf-8")

    slices = {d.name: {"name": d.name, "shape": list(d.keypoints.shape), "fps": d.fps,
                       "has_labels": d.labels is not None, "notes": d.notes} for d in datasets}
    slices_path = OUT_DIR / "dataset_slices.json"
    if slices_path.exists():
        slices = {s["name"]: s for s in json.loads(slices_path.read_text())} | slices
    slices_path.write_text(json.dumps(list(slices.values()), indent=2), encoding="utf-8")
    plot_summary(df)
    write_html(df, list(slices.values()))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
