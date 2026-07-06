"""Complete the 7-keypoint noise-injection sweep started manually for
tail_base and neck (260706_behaviorlab_calms21_stgcn_gradcam.md).

Reuses train_calms21_stgcn.py / gradcam_calms21_stgcn.py / make_tailbase_noised_calms21.py
directly (no duplicated training/Grad-CAM logic) -- just loops the same
data-prep -> train -> Grad-CAM pipeline over the remaining keypoints so the
"any keypoint noise -> F1 up + importance shifts to whatever's still clean"
pattern can be checked for generality rather than asserted from n=2.

Usage:
    python scripts/keypoint_noise_sweep.py
"""
import argparse
import os, sys
sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from make_tailbase_noised_calms21 import add_noise, JOINT_NAMES, SRC
from train_calms21_stgcn import build_loaders, evaluate, majority_baseline_f1
from gradcam_calms21_stgcn import per_class_importance
from behavior_lab.data.loaders.calms21 import CLASS_NAMES
from behavior_lab.models.graph.baselines import STGCN
from behavior_lab.training import Trainer

REMAINING = ["left_ear", "right_ear", "left_hip", "right_hip"]  # nose already done (F1=0.4327, see research note)
EPOCHS = 8


def make_noised_npz(keypoint: str) -> str:
    dst = f"data/calms21/calms21_aligned_{keypoint}_noised.npz"
    rng = np.random.default_rng(0)
    d = np.load(SRC, allow_pickle=True)
    idx = JOINT_NAMES.index(keypoint)
    np.savez(dst, x_train=add_noise(d["x_train"], idx, rng), y_train=d["y_train"],
              x_test=add_noise(d["x_test"], idx, rng), y_test=d["y_test"])
    return dst


def run_one(keypoint: str, seed: int = 42) -> dict:
    data_path = f"data/calms21/calms21_aligned_{keypoint}_noised.npz"
    if not os.path.exists(data_path):
        data_path = make_noised_npz(keypoint)
    output_dir = f"outputs/calms21_stgcn_{keypoint}_noised" + (f"_seed{seed}" if seed != 42 else "")
    _, _, train_loader, test_loader = build_loaders(False, 32, 4000, 1200, data_path)

    model = STGCN(num_classes=4, num_joints=7, num_persons=2, in_channels=2, skeleton="calms21")
    trainer = Trainer(model, train_loader, test_loader, cfg={
        "device": "cpu", "num_epoch": EPOCHS, "batch_size": 32,
        "output_dir": output_dir, "patience": EPOCHS, "seed": seed,
        "learning_rate": 0.1, "warmup_epochs": 1, "momentum": 0.9, "nesterov": True,
        "weight_decay": 0.0004, "lr_decay_rate": 0.1,
        "lr_steps": [max(2, EPOCHS - 4), max(3, EPOCHS - 2)],
    })
    trainer.train()

    metrics, test_labels = evaluate(model, test_loader, torch.device("cpu"))
    baseline_f1 = majority_baseline_f1(test_labels)

    importance, counts = per_class_importance(model, test_loader)
    top_per_class = {CLASS_NAMES[c]: JOINT_NAMES[imp.mean(axis=0).argmax()] for c, imp in importance.items()}

    return {
        "keypoint": keypoint, "f1_macro": metrics.f1_macro, "baseline_f1": baseline_f1,
        "signal": metrics.f1_macro - baseline_f1, "top_per_class": top_per_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint", type=str, default=None, choices=JOINT_NAMES,
                     help="Run exactly one keypoint (each full train+Grad-CAM pass takes "
                          "~25-30min -- looping several in one process risks getting killed "
                          "mid-run, see Next in the research note. Always pass this one at a time.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    keypoints = [args.keypoint] if args.keypoint else REMAINING

    results = []
    for kp in keypoints:
        print(f"\n=== noising {kp} (seed={args.seed}) ===")
        r = run_one(kp, seed=args.seed)
        results.append(r)
        print(f"{kp}: F1-macro={r['f1_macro']:.4f} (baseline {r['baseline_f1']:.4f}, "
              f"signal {r['signal']:+.4f}) top-per-class={r['top_per_class']}")

    print("\n| Noised keypoint | F1-macro | vs clean (0.3053) | top keypoint (attack/mount/investigation/other) |")
    print("|---|---:|---:|---|")
    for r in results:
        tp = r["top_per_class"]
        tops = "/".join(tp.get(c, "-") for c in ["attack", "mount", "investigation", "other"])
        print(f"| {r['keypoint']} | {r['f1_macro']:.4f} | {r['f1_macro']-0.3053:+.4f} | {tops} |")


if __name__ == "__main__":
    main()
