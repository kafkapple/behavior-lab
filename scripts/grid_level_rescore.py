"""Re-score the discovery grid at matched hierarchy levels from stored labels (no model fit).

The grid compared SUBTLE's finest supercluster (3-5 clusters) with keypoint-MoSeq syllables (20),
so the levels differed. This script compares, on the same stored outputs:
  old    long run, SUBTLE = finest supercluster (what the grid reported)
  fine   long run, SUBTLE = subclusters (Phenograph level), every other method at its own fine setting
  coarse matched5 run, SUBTLE = supercluster level whose count is closest to 5
Every number comes from ``label_agreement`` (ARI, AMI, circular-shift null); nothing is fitted.

Usage: ``uv run python scripts/grid_level_rescore.py [--out outputs/behavior_analysis_workbench/level_rescore]``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from behavior_lab.visualization.agreement import label_agreement

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "behavior_analysis_workbench"
# slices whose SUBTLE sub/supercluster labels were stored (subtle_map_extra.npz); the other recordings kept the final level only
SLICES = ["subtle_6112", "shank3ko_KO_100", "avatar_st_optim"]
OTHERS = ["b_soid", "keypoint_moseq", "kmeans_pca_umap", "pca_hmm_moseq_fallback"]


def load(run: str, sl: str, method: str) -> np.ndarray | None:
    f = BASE / run / "arrays" / sl / f"{method}_labels.npy"
    return np.load(f) if f.exists() else None  # a method can be missing on a slice (not run or not applicable)


def subtle_levels(run: str, sl: str) -> dict[str, np.ndarray]:
    z = np.load(BASE / run / "arrays" / sl / "subtle_map_extra.npz", allow_pickle=True)
    sup = z["superclusters"]  # (T, n_levels), coarse to fine as stored
    counts = [len(set(sup[:, k].tolist())) for k in range(sup.shape[1])]
    return {"sub": z["subclusters"], "super": sup, "counts": np.array(counts), "final": z["labels"]}


def summarize(seqs: dict[str, np.ndarray]) -> dict[str, object]:
    ag = label_agreement(seqs, n_shifts=20, seed=0)
    n = len(ag["names"])
    iu = np.triu_indices(n, 1)
    top = {k: float(np.bincount(np.unique(v, return_inverse=True)[1]).max() / len(v)) for k, v in seqs.items()}
    return {
        "names": ag["names"], "k": {k: int(len(set(v.tolist()))) for k, v in seqs.items()},
        "ari_median": float(np.median(ag["ari"][iu])), "ami_median": float(np.median(ag["ami"][iu])),
        "null_ari_median": float(np.median(ag["null_ari"][iu])), "null_ami_median": float(np.median(ag["null_ami"][iu])),
        "top_share": top, "ari": ag["ari"].tolist(), "ami": ag["ami"].tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "level_rescore"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict[str, dict] = {}
    for sl in SLICES:
        lv_long, lv_m5 = subtle_levels("long", sl), subtle_levels("matched5", sl)
        T = len(lv_long["sub"])
        assert T == len(load("long", sl, "kmeans_pca_umap")), f"{sl}: frame count mismatch"
        k5 = int(np.argmin(np.abs(lv_m5["counts"] - 5)))
        conds = {
            "old": {"subtle": lv_long["final"], **{m: load("long", sl, m) for m in OTHERS}},
            "fine": {"subtle_sub": lv_long["sub"], **{m: load("long", sl, m) for m in OTHERS}},
            "coarse": {"subtle_super": lv_m5["super"][:, k5], **{m: load("matched5", sl, m) for m in OTHERS}},
        }
        conds = {c: {k: v for k, v in s.items() if v is not None} for c, s in conds.items()}
        res[sl] = {c: summarize(s) for c, s in conds.items()}
        res[sl]["subtle_super_counts_long"] = lv_long["counts"].tolist()
        res[sl]["coarse_super_level"] = k5
        print(sl, {c: (round(res[sl][c]["ari_median"], 3), round(res[sl][c]["ami_median"], 3)) for c in conds},
              "k_fine", res[sl]["fine"]["k"])
    summary = {}
    for c in ("old", "fine", "coarse"):
        summary[c] = {key: float(np.median([res[s][c][key] for s in SLICES]))
                      for key in ("ari_median", "ami_median", "null_ari_median", "null_ami_median")}
    # self-check: the old condition must reproduce the reported grid scale (ARI median about 0.07)
    assert 0.0 <= summary["old"]["ari_median"] <= 0.2, summary["old"]
    json.dump({"slices": res, "median_over_slices": summary}, open(out / "level_rescore.json", "w"), indent=1)
    print("median over slices:", json.dumps(summary))


if __name__ == "__main__":
    main()
