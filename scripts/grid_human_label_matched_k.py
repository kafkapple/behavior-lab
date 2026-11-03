"""Re-score the discovery grid against human labels at matched cluster counts (stored labels, no fit).

Follow-up of ``grid_human_label_score.py``. A devil review listed three confounds of its AMI table:
cluster counts differ (SUBTLE 3, B-SOiD 41), ``na`` frames are dropped, and AMI depends on k. Here
  coarsen    keep the k-1 largest clusters of a stored label, merge the rest into one (k = 4, 5, 8);
             a label with fewer clusters is left as is
  ami        AMI vs the 4 human classes on annotated frames (as before)
  ami_na     AMI vs 5 classes, ``na`` kept as a class, all frames
  bacc       balanced accuracy of a cluster -> majority-class map learned on the first half of the
             frames and tested on the second half (annotated frames only); cluster ids are not
             shared across recordings, so the map cannot be learned across recordings
Rule declared before the run: method X is ahead of Y when X > Y in at least 12 of the 15 paired
(recording, seed) cells (one-sided binomial p < 0.02; seeds are not independent, so it is lenient).

Usage: ``uv run python scripts/grid_human_label_matched_k.py``
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, balanced_accuracy_score

from behavior_lab.visualization.agreement import stretch_labels
from grid_human_label_score import ANNOTS, BASE, METHODS, SLICES, human_labels

KS, SEEDS, WINS = (0, 4, 5, 8), (42, 43, 44), 12  # k = 0: stored labels, no coarsening


def coarsen(lab: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return np.unique(lab, return_inverse=True)[1]
    ids, counts = np.unique(lab, return_counts=True)
    if len(ids) <= k:
        return np.unique(lab, return_inverse=True)[1]
    top = ids[np.argsort(-counts)[: k - 1]]
    return np.where(np.isin(lab, top), np.searchsorted(np.sort(top), lab), k - 1)


def bacc(lab: np.ndarray, hum: np.ndarray) -> float:
    half = len(lab) // 2
    fit = hum[:half] >= 0
    default = np.bincount(hum[:half][fit]).argmax()
    cmap = {c: np.bincount(hum[:half][fit & (lab[:half] == c)], minlength=4).argmax()
            for c in np.unique(lab[:half]) if (fit & (lab[:half] == c)).any()}
    test = hum[half:] >= 0
    pred = np.array([cmap.get(c, default) for c in lab[half:][test]])
    return float(balanced_accuracy_score(hum[half:][test], pred))


def scores(lab: np.ndarray, hum: np.ndarray) -> dict[str, float]:
    keep = hum >= 0
    hum5 = np.where(keep, hum, 4)
    return {"ami": float(adjusted_mutual_info_score(hum[keep], lab[keep])),
            "ami_na": float(adjusted_mutual_info_score(hum5, lab)), "bacc": bacc(lab, hum)}


def main() -> None:
    cells: dict[int, dict[str, list[dict]]] = {k: {m: [] for m in METHODS} for k in KS}
    for sl, sess in SLICES.items():
        hum = human_labels(ANNOTS / f"{sess}.csv")
        for m in METHODS:
            for seed in SEEDS:
                f = BASE / "long" / "arrays" / sl / "repeats" / f"{m}_seed{seed}.npy"
                if not f.exists():
                    continue
                lab = stretch_labels(np.load(f), len(hum))
                for k in KS:
                    c = coarsen(lab, k)
                    assert k == 0 or len(set(c.tolist())) <= k
                    cells[k][m].append({"slice": sl, "seed": seed, "k": int(len(set(c.tolist()))), **scores(c, hum)})
    # self-check: random labels score about zero AMI and chance balanced accuracy (4 classes, about 0.25)
    rng = np.random.default_rng(1)
    hum = human_labels(ANNOTS / "adult_6112.csv")
    ctrl = scores(rng.integers(0, 5, size=len(hum)), hum)
    assert abs(ctrl["ami"]) < 0.01 and abs(ctrl["ami_na"]) < 0.01 and abs(ctrl["bacc"] - 0.25) < 0.1, ctrl
    out: dict = {"control": ctrl, "rule": f">= {WINS} of 15 paired cells", "by_k": {}}
    for k in KS:
        out["by_k"][str(k)] = {}
        for metric in ("ami", "ami_na", "bacc"):
            summ = {}
            for m in METHODS:
                v = np.array([c[metric] for c in cells[k][m]])
                summ[m] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                           "k_median": float(np.median([c["k"] for c in cells[k][m]]))}
            ahead, ahead_rec = [], []
            for x, y in combinations(METHODS, 2):
                vx = np.array([c[metric] for c in cells[k][x]])
                vy = np.array([c[metric] for c in cells[k][y]])
                assert len(vx) == len(vy) == 15, (len(vx), len(vy))
                # recording-level check: the 3 seeds of a recording are not independent
                rx = np.median(vx.reshape(5, 3), axis=1); ry = np.median(vy.reshape(5, 3), axis=1)
                rec = int((rx > ry).sum())
                if rec == 5 or rec == 0:
                    ahead_rec.append(f"{x if rec == 5 else y} > {y if rec == 5 else x} (5/5 recordings)")
                w = int((vx > vy).sum())
                if w >= WINS:
                    ahead.append(f"{x} > {y} ({w}/15)")
                elif 15 - w >= WINS:
                    ahead.append(f"{y} > {x} ({15 - w}/15)")
            out["by_k"][str(k)][metric] = {"summary": summ, "ahead": ahead, "ahead_5of5_recordings": ahead_rec}
    Path(BASE / "level_rescore").mkdir(exist_ok=True)
    (BASE / "level_rescore" / "human_label_matched_k.json").write_text(json.dumps(out, indent=1))
    print("random control", {a: round(b, 3) for a, b in ctrl.items()})
    for k in KS:
        for metric in ("ami", "ami_na", "bacc"):
            r = out["by_k"][str(k)][metric]
            print(f"k={k} {metric}: " + "  ".join(f"{m[:8]} {v['median']:.3f}[{v['min']:.2f},{v['max']:.2f}]" for m, v in r["summary"].items()))
            print("   ahead:", r["ahead"] or "none")
            print("   5/5 recordings:", r["ahead_5of5_recordings"] or "none")


if __name__ == "__main__":
    main()
