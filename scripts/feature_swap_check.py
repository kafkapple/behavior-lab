"""Feature swap 2x2 on stored embeddings: is label disagreement a feature effect or a model effect?

k-means (k=5, best of 5 seeds) is fitted on the 2D embedding each method stored.
  A = k-means(embedding X) vs the method's own label X   (same feature, different model)
  B = k-means(embedding X) vs k-means(embedding Y)        (different feature, same model)
Verdict rule declared before the run in the vault note 261002_behaviorlab_discovery_grid_long_run
(reused from 261002_behaviorlab_feature_control): A >= 2B and A - B >= 0.10 -> feature effect;
B >= 2A and B - A >= 0.10 -> model effect; else indistinct. Limit: the embedding is a 2D
projection, not the method's original feature; keypoint-MoSeq stores none and is left out.

Usage: ``uv run python scripts/feature_swap_check.py``
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from behavior_lab.visualization.agreement import shift_within, stretch_labels

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "behavior_analysis_workbench"
SLICES = ["subtle_6112", "shank3ko_KO_100", "avatar_st_optim"]
METHODS = ["subtle", "b_soid", "kmeans_pca_umap", "pca_hmm_moseq_fallback"]
K, SEEDS, N_SHIFTS, RATIO, GAP = 5, range(5), 10, 2.0, 0.10


def kmeans_labels(emb: np.ndarray) -> np.ndarray:
    fits = [KMeans(K, n_init=1, random_state=s).fit(emb) for s in SEEDS]
    return min(fits, key=lambda m: m.inertia_).labels_


def main() -> None:
    rng = np.random.default_rng(0)
    shifts = rng.uniform(0.1, 0.9, size=N_SHIFTS)
    rows = []
    for sl in SLICES:
        d = BASE / "long" / "arrays" / sl
        emb = {m: np.load(d / f"{m}_embedding.npy") for m in METHODS}
        lab = {m: np.load(d / f"{m}_labels.npy") for m in METHODS}
        T = max(len(v) for v in lab.values())
        km = {m: stretch_labels(kmeans_labels(emb[m]), T) for m in METHODS}
        own = {m: stretch_labels(lab[m], T) for m in METHODS}
        assert all(len(v) == T for v in list(km.values()) + list(own.values())), "length mismatch"
        for m in METHODS:
            a = float(adjusted_rand_score(km[m], own[m]))
            null = float(np.mean([adjusted_rand_score(km[m], shift_within(own[m], s)) for s in shifts]))
            rows.append({"slice": sl, "kind": "A", "pair": [m, m], "ari": a, "null": null})
        for x, y in combinations(METHODS, 2):
            b = float(adjusted_rand_score(km[x], km[y]))
            null = float(np.mean([adjusted_rand_score(km[x], shift_within(km[y], s)) for s in shifts]))
            rows.append({"slice": sl, "kind": "B", "pair": [x, y], "ari": b, "null": null})
    A = float(np.median([r["ari"] for r in rows if r["kind"] == "A"]))
    B = float(np.median([r["ari"] for r in rows if r["kind"] == "B"]))
    verdict = ("feature effect dominates" if A >= RATIO * B and A - B >= GAP else
               "model effect dominates" if B >= RATIO * A and B - A >= GAP else "indistinct")
    per_slice = {sl: {k: float(np.median([r["ari"] for r in rows if r["slice"] == sl and r["kind"] == k]))
                      for k in "AB"} for sl in SLICES}
    assert -0.1 <= A <= 1 and -0.1 <= B <= 1, (A, B)
    out = BASE / "feature_control"
    out.mkdir(exist_ok=True)
    (out / "feature_swap.json").write_text(json.dumps(
        {"k": K, "A_median": A, "B_median": B, "verdict": verdict, "per_slice": per_slice, "rows": rows}, indent=1))
    print(f"A (same feature, other model) median ARI {A:.3f}; B (other feature, same model) {B:.3f}; "
          f"verdict: {verdict}\nper slice: {json.dumps(per_slice)}")
    for r in rows:
        if r["kind"] == "A":
            print(f"  A {r['slice']:16s} {r['pair'][0]:24s} ARI {r['ari']:.3f} null {r['null']:.3f}")


if __name__ == "__main__":
    main()
