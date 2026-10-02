"""Feature control: do k-means and the HMM disagree because of their inputs or their models?

3 feature sets x 3 models on the same frames (the pooled SUBTLE recordings of a finished
batch), every model fed the same standardized matrix. Reports ARI between cells, the same
divided by sqrt(repeat ARI product), a within-recording shifted null, and the pre-declared
verdict (vault note 261002_behaviorlab_feature_control).

    uv run python scripts/feature_control_check.py [--batch long] [--slice subtle_pooled]
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from behavior_lab.data.features import SkeletonBackend
from behavior_lab.visualization.agreement import shift_within

ROOT = Path(__file__).resolve().parents[1]
K, SEEDS, N_SHIFTS = 5, range(10), 20
RATIO, GAP = 2.0, 0.10  # verdict thresholds, fixed before the run


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)


def egocentric(kp: np.ndarray, front: int, back: int) -> np.ndarray:
    """Remove the horizontal position and heading of every frame (height is kept)."""
    out = kp.astype(float).copy()
    out[:, :, :2] -= out[:, :, :2].mean(axis=1, keepdims=True)
    v = out[:, front, :2] - out[:, back, :2]
    ang = np.arctan2(v[:, 1], v[:, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    x, y = out[:, :, 0].copy(), out[:, :, 1].copy()
    out[:, :, 0] = c[:, None] * x - s[:, None] * y
    out[:, :, 1] = s[:, None] * x + c[:, None] * y
    v2 = out[:, front, :2] - out[:, back, :2]
    assert np.abs(v2[:, 1]).max() < 1e-3 * (np.abs(v2[:, 0]).max() + 1e-9) + 1e-6, "not aligned"
    return out


def features(kp: np.ndarray, lengths: list[int], fps: float, names: list[str]) -> dict:
    segs = np.split(kp, np.cumsum(lengths)[:-1])
    backend = SkeletonBackend(fps=fps, normalize_body_size=True)
    ego = egocentric(kp, names.index("neck"), names.index("tail_base"))
    return {
        "summary4": standardize(np.concatenate([backend.extract(s) for s in segs])),
        "raw_pca10": standardize(PCA(10, random_state=0).fit_transform(kp.reshape(len(kp), -1))),
        "ego_pca10": standardize(PCA(10, random_state=0).fit_transform(ego.reshape(len(kp), -1))),
    }


def fit(model: str, X: np.ndarray, lengths: list[int], seed: int) -> tuple[np.ndarray, float]:
    """Labels and a fit score (higher = better) for choosing the representative seed."""
    if model == "kmeans":
        m = KMeans(K, n_init=1, random_state=seed).fit(X)
        return m.labels_, -float(m.inertia_)
    if model == "gmm":
        m = GaussianMixture(K, covariance_type="diag", random_state=seed).fit(X)
        return m.predict(X), float(m.score(X))
    from hmmlearn.hmm import GaussianHMM

    m = GaussianHMM(K, covariance_type="diag", n_iter=50, random_state=seed).fit(X, lengths)
    return m.predict(X, lengths), float(m.score(X, lengths))


def median_bout_sec(labels: np.ndarray, fps: float) -> float:
    runs = np.diff(np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True]))
    return float(np.median(runs) / fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="long")
    ap.add_argument("--slice", default="subtle_pooled")
    args = ap.parse_args()
    batch = ROOT / "outputs" / "behavior_analysis_workbench" / args.batch
    sl = next(s for s in json.loads((batch / "dataset_slices.json").read_text())
              if s["name"] == args.slice)
    kp = np.load(batch / "arrays" / args.slice / "keypoints.npy")
    lengths, fps = sl["notes"]["lengths"], sl["fps"]
    assert sum(lengths) == len(kp)
    feats = features(kp, lengths, fps, sl["notes"]["node_names"])

    cells: dict[tuple[str, str], dict] = {}
    for fname, X in feats.items():
        for model in ("kmeans", "gmm", "hmm"):
            runs = [fit(model, X, lengths, s) for s in SEEDS]
            labs = [r[0] for r in runs]
            rep = float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labs, 2)]))
            best = labs[int(np.argmax([r[1] for r in runs]))]
            cells[(fname, model)] = {"labels": best, "repeat_ari": rep,
                                     "median_bout_sec": median_bout_sec(best, fps)}
            print(f"{fname:10s} {model:6s} repeat ARI {rep:.2f}  median bout "
                  f"{cells[(fname, model)]['median_bout_sec']:.2f} s", flush=True)

    rng = np.random.default_rng(0)
    shifts = rng.uniform(0.1, 0.9, size=N_SHIFTS)
    pairs = []
    for a, b in combinations(cells, 2):
        la, lb = cells[a]["labels"], cells[b]["labels"]
        ari = float(adjusted_rand_score(la, lb))
        null = float(np.mean([adjusted_rand_score(la, shift_within(lb, s, lengths))
                              for s in shifts]))
        ceil = float(np.sqrt(max(cells[a]["repeat_ari"], 0) * max(cells[b]["repeat_ari"], 0)))
        kind = ("same_feature" if a[0] == b[0] else "same_model" if a[1] == b[1] else "neither")
        pairs.append({"a": list(a), "b": list(b), "kind": kind, "ari": ari, "null": null,
                      "normalized": ari / ceil if ceil > 0 else None})
    A = float(np.median([p["normalized"] for p in pairs if p["kind"] == "same_feature"]))
    B = float(np.median([p["normalized"] for p in pairs if p["kind"] == "same_model"]))
    verdict = ("feature effect dominates" if A >= RATIO * B and A - B >= GAP else
               "model effect dominates" if B >= RATIO * A and B - A >= GAP else "indistinct")
    out = batch.parent / "feature_control"
    out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps({
        "slice": args.slice, "k": K, "seeds": len(SEEDS),
        "cells": [{"feature": f, "model": m, "repeat_ari": c["repeat_ari"],
                   "median_bout_sec": c["median_bout_sec"]} for (f, m), c in cells.items()],
        "pairs": pairs, "A_same_feature": A, "B_same_model": B, "verdict": verdict}, indent=2))
    for p in sorted(pairs, key=lambda p: (p["kind"], -p["ari"])):
        if p["kind"] != "neither":
            print(f"{p['kind']:12s} {'/'.join(p['a']):18s} x {'/'.join(p['b']):18s} "
                  f"ARI {p['ari']:.3f} null {p['null']:.3f} normalized {p['normalized']:.2f}")
    print(f"A (same feature, other model) = {A:.2f}; B (same model, other feature) = {B:.2f}; "
          f"verdict: {verdict}")


if __name__ == "__main__":
    main()
