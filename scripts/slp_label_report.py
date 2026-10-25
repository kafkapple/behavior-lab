"""Check a SLEAP label set shipped with a model (coordinates only) and write the dashboard page.

    python scripts/slp_label_report.py --model-dir <dir with labels_gt.train.0.slp, labels_gt.val.0.slp[, labels_pr.val.0.slp, metrics.val.0.npz]> \
        --out training_labels.html --provenance "path or note" --route "Get the .pkg.slp=not found yet"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from behavior_lab.pose import slp_labels, slp_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analysis-json", type=Path, help="also write the analysis here")
    ap.add_argument("--provenance", action="append", default=[])
    ap.add_argument("--route", action="append", default=[], help="'route=state' rows of the image problem table")
    ap = ap.parse_args()
    d = ap.model_dir
    train, val = slp_labels.load_slp(d / "labels_gt.train.0.slp"), slp_labels.load_slp(d / "labels_gt.val.0.slp")
    pred = slp_labels.load_predictions(d / "labels_pr.val.0.slp") if (d / "labels_pr.val.0.slp").exists() else None
    a = slp_labels.analyze(train, val, pred)
    metrics = None
    if (d / "metrics.val.0.npz").exists():
        m = np.load(d / "metrics.val.0.npz", allow_pickle=True)["metrics"].item()
        metrics = {"mOKS": round(float(m["mOKS"]["mOKS"]), 3), "mPCK": round(float(m["pck_metrics"]["mPCK"]), 3),
                   "val distance p50 / p90 px": f"{m['distance_metrics']['p50']:.1f} / {m['distance_metrics']['p90']:.1f}"}
    if ap.analysis_json:
        ap.analysis_json.write_text(json.dumps(a))
    print(slp_report.build(a, ap.out, ap.provenance, [tuple(r.split("=", 1)) for r in ap.route], metrics))


if __name__ == "__main__":
    main()
