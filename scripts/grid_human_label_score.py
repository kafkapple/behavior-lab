"""Score the discovery grid against the public human labels of the SUBTLE y5a5 recordings.

The five SUBTLE slices of the grid (adult_6112, 6115, 6116, 6127, 7678) have per-frame human
labels in the public SUBTLE dataset (``annots/adult_<id>.csv``: grooming, rearing, standing,
walking, na = not annotated). Every method's stored label is compared with them on the
annotated frames only (na dropped). Nothing is fitted.

  ARI, AMI    chance-corrected agreement with the 4 human classes
  homogeneity each cluster holds one human class (not hurt by over-segmentation)
  null        the same score after circularly shifting the method label (20 shifts)
The human classes are coarse, so a method that splits finer scores low on ARI/AMI.

Usage: ``uv run python scripts/grid_human_label_score.py [--annots DIR]``
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score

from behavior_lab.visualization.agreement import shift_within, stretch_labels

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "behavior_analysis_workbench"
ANNOTS = Path.home() / "dev" / "external" / "subtle_ref" / "subtle" / "dataset" / "y5a5" / "annots"
SLICES = {"subtle_6112": "adult_6112", "subtle_6115": "adult_6115", "subtle_6116": "adult_6116",
          "subtle_6127": "adult_6127", "subtle_7678": "adult_7678"}
METHODS = ["b_soid", "keypoint_moseq", "kmeans_pca_umap", "pca_hmm_moseq_fallback", "subtle"]
RUNS = ["long", "matched5"]
N_SHIFTS = 20


def human_labels(path: Path) -> np.ndarray:
    rows = list(csv.reader(open(path)))[1:]
    names = sorted({r[1] for r in rows} - {"na"})
    code = {n: i for i, n in enumerate(names)}
    code["na"] = -1
    return np.array([code[r[1]] for r in rows])


def score(lab: np.ndarray, hum: np.ndarray, shifts: np.ndarray) -> dict[str, float]:
    keep = hum >= 0
    h = hum[keep]

    def three(x: np.ndarray) -> tuple[float, float, float]:
        x = x[keep]
        return (adjusted_rand_score(h, x), adjusted_mutual_info_score(h, x), homogeneity_score(h, x))

    ari, ami, hom = three(lab)
    nulls = np.array([three(shift_within(lab, s)) for s in shifts])
    return {"ari": ari, "ami": ami, "hom": hom, "ari_null": float(nulls[:, 0].mean()),
            "ami_null": float(nulls[:, 1].mean()), "hom_null": float(nulls[:, 2].mean()),
            "k": int(len(set(lab[keep].tolist())))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annots", default=str(ANNOTS))
    annots = Path(ap.parse_args().annots)
    shifts = np.random.default_rng(0).uniform(0.1, 0.9, size=N_SHIFTS)
    res: dict[str, dict] = {}
    for run in RUNS:
        for sl, sess in SLICES.items():
            hum = human_labels(annots / f"{sess}.csv")
            for m in METHODS:
                f = BASE / run / "arrays" / sl / f"{m}_labels.npy"
                if not f.exists():
                    continue
                lab = stretch_labels(np.load(f), len(hum))
                assert len(lab) == len(hum), f"{sl}: label {len(lab)} vs human {len(hum)}"
                res.setdefault(run, {}).setdefault(m, {})[sl] = score(lab, hum, shifts)
    summary = {run: {m: {k: float(np.median([v[k] for v in per.values()]))
                         for k in ("ari", "ami", "hom", "ami_null", "k")}
                     for m, per in ms.items()} for run, ms in res.items()}
    # seed spread (long run, 3 repeats): the stored label above is one seed, and SUBTLE is unseeded
    spread: dict[str, dict] = {}
    for m in METHODS:
        vals = []
        for sl, sess in SLICES.items():
            hum = human_labels(annots / f"{sess}.csv")
            for seed in (42, 43, 44):
                f = BASE / "long" / "arrays" / sl / "repeats" / f"{m}_seed{seed}.npy"
                if f.exists():
                    lab = stretch_labels(np.load(f), len(hum))
                    keep = hum >= 0
                    vals.append((sl, seed, float(adjusted_mutual_info_score(hum[keep], lab[keep]))))
        a = np.array([v[2] for v in vals])
        wins = {sl: float(np.median([v[2] for v in vals if v[0] == sl])) for sl in SLICES}
        spread[m] = {"ami_median": float(np.median(a)), "ami_min": float(a.min()), "ami_max": float(a.max()),
                     "n": len(a), "per_slice_median": wins}
    best = {sl: max(spread, key=lambda m: spread[m]["per_slice_median"][sl]) for sl in SLICES}
    # self-check: chance level. A random-label control must score about zero ARI/AMI
    rng = np.random.default_rng(1)
    hum = human_labels(annots / "adult_6112.csv")
    ctrl = score(rng.integers(0, 5, size=len(hum)), hum, shifts)
    assert abs(ctrl["ari"]) < 0.01 and abs(ctrl["ami"]) < 0.01, ctrl
    out = BASE / "level_rescore"
    out.mkdir(exist_ok=True)
    (out / "human_label_score.json").write_text(json.dumps({"summary": summary, "slices": res, "seed_spread": spread, "best_per_slice": best}, indent=1))
    print("seed spread (long, 3 repeats x 5 slices), human AMI median [min, max]:")
    for m, v in spread.items():
        print(f"  {m:24s} {v['ami_median']:.3f} [{v['ami_min']:.3f}, {v['ami_max']:.3f}] n={v['n']}")
    print("best method per slice (median over 3 seeds):", best)
    for run, ms in summary.items():
        print(run)
        for m, v in sorted(ms.items(), key=lambda kv: -kv[1]["ami"]):
            print(f"  {m:24s} k={v['k']:5.0f} ARI {v['ari']:.3f} AMI {v['ami']:.3f} (null {v['ami_null']:.3f}) hom {v['hom']:.3f}")


if __name__ == "__main__":
    main()
