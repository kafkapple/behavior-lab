"""Disambiguation probe: is a keypoint's Grad-CAM dominance real signal or a
"cleanest-tracked-coordinate" artifact?

Injects Gaussian noise (Nx that keypoint's own natural std) into one named
keypoint for both mice, in both train and test splits, and saves a new npz.
Re-running train_calms21_stgcn.py / gradcam_calms21_stgcn.py against this file
(same seeds, same subsample) isolates the effect of degrading just that one
keypoint's signal quality.

Includes a control: noising a DIFFERENT keypoint (e.g. neck) checks whether any
F1 change is specific to the targeted keypoint or just a generic
noise-as-regularization effect on a small training subsample -- if noising any
keypoint improves F1 similarly, the "shortcut removal" story doesn't hold.

Usage:
    python scripts/make_tailbase_noised_calms21.py --keypoint tail_base
    python scripts/make_tailbase_noised_calms21.py --keypoint neck   # control
"""
import argparse
import numpy as np

SRC = "data/calms21/calms21_aligned.npz"
JOINT_NAMES = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]
NOISE_MULT = 3.0


def add_noise(x: np.ndarray, joint_idx: int, rng: np.random.Generator) -> np.ndarray:
    """x: (N, 2 mice, T, 7, 2). Adds noise to both mice's coords for one joint."""
    x = x.copy()
    jp = x[:, :, :, joint_idx, :]           # (N, 2, T, 2)
    sigma = jp.std()
    x[:, :, :, joint_idx, :] = jp + rng.normal(0, NOISE_MULT * sigma, jp.shape)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint", type=str, default="tail_base", choices=JOINT_NAMES)
    args = ap.parse_args()
    joint_idx = JOINT_NAMES.index(args.keypoint)
    dst = f"data/calms21/calms21_aligned_{args.keypoint}_noised.npz"

    rng = np.random.default_rng(0)
    d = np.load(SRC, allow_pickle=True)
    x_train_noised = add_noise(d["x_train"], joint_idx, rng)
    x_test_noised = add_noise(d["x_test"], joint_idx, rng)
    np.savez(dst, x_train=x_train_noised, y_train=d["y_train"],
              x_test=x_test_noised, y_test=d["y_test"])
    print(f"Saved {dst} ({args.keypoint} noise = {NOISE_MULT}x natural std, both mice, train+test)")


if __name__ == "__main__":
    main()
