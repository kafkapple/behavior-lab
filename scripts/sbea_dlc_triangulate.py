"""SBeA: author DLC model → 2D per camera → DLT triangulation → (T, K, 3) npz.

SBeA ships video + P-matrix calibration + a trained DLC snapshot, but no 3D
keypoints. This produces them, in behavior-lab's canonical `(T, K, D)` layout
(spec: docs/architecture.md §Data Format Specification).

The snapshot ships without pose_cfg.yaml, so it is rebuilt from the checkpoint
itself: backbone scope gives net_type, pose/part_pred/block4/biases gives
num_joints. Joint order follows the author's released CSV header.

Single-animal only — the released model is `Mouse2Dproject` (one pose per
frame). Social sessions need identity separation first, so point --session at
an `individual/` recording.

Env: conda `dlc2` (DLC 2.3.11 + TF 2.12.1). `dlc3` has no TF backend.

Usage:
    /home/joon/anaconda3/envs/dlc2/bin/python scripts/sbea_dlc_triangulate.py \\
        --root /node_data_2/joon/data/external/SBeA/individual \\
        --session rec10-M2-20221108 --end 600 --step 3 \\
        --out /node_data/joon/data/sbea_kp3d
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importable either way
import sbea_lr_resolve as lr  # noqa: E402

BODYPARTS = ["nose", "left_ear", "right_ear", "neck", "left_front_limb",
             "right_front_limb", "left_hind_limb", "right_hind_limb",
             "left_front_claw", "right_front_claw", "left_hind_claw",
             "right_hind_claw", "back", "root_tail", "mid_tail", "tip_tail"]
N_CAMS = 4


def build_pose_cfg(snapshot: str, num_joints: int):
    """DLC's bundled template + the two fields the snapshot determines."""
    import deeplabcut.pose_estimation_tensorflow as pet
    from deeplabcut.pose_estimation_tensorflow.config import load_config

    tmpl = Path(pet.__file__).resolve().parent.parent / "pose_cfg.yaml"
    cfg = load_config(str(tmpl))
    cfg.update({
        "net_type": "resnet_50", "num_joints": num_joints,
        "all_joints": [[i] for i in range(num_joints)],
        "all_joints_names": BODYPARTS[:num_joints],
        "init_weights": snapshot, "batch_size": 1,
        "location_refinement": True, "locref_stdev": 7.2801,
    })
    return cfg


def infer_camera(video: Path, frames: list[int], cfg, sess, inputs, outputs) -> np.ndarray:
    """(len(frames), K, 3) — x, y, likelihood. Sequential read; videos are long."""
    from deeplabcut.pose_estimation_tensorflow.core import predict

    cap = cv2.VideoCapture(str(video))
    out = np.full((len(frames), cfg["num_joints"], 3), np.nan)
    want = {f: i for i, f in enumerate(frames)}
    fi, seen = 0, 0
    while seen < len(frames):
        ok, bgr = cap.read()
        if not ok:
            break
        if fi in want:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out[want[fi]] = predict.getpose(rgb, cfg, sess, inputs, outputs)
            seen += 1
        fi += 1
    cap.release()
    if seen < len(frames):
        print(f"  ! {video.name}: {seen}/{len(frames)} frames read")
    return out


def triangulate(pts2d: np.ndarray, probs: np.ndarray, Ps: list[np.ndarray],
                prob_min: float) -> np.ndarray:
    """Linear DLT over the views that pass prob_min. NaN if fewer than 2."""
    rows = []
    for P, (x, y), q in zip(Ps, pts2d, probs):
        if not np.isfinite([x, y]).all() or q < prob_min:
            continue
        rows += [x * P[2] - P[0], y * P[2] - P[1]]
    if len(rows) < 4:
        return np.full(3, np.nan)
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-9 else np.full(3, np.nan)


def reproject(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2] if abs(h[2]) > 1e-9 else np.full(2, np.nan)


def median_residual(kp2d: np.ndarray, Plist: list[np.ndarray], prob_min: float,
                    n_t: int = 30) -> float:
    res = []
    for t in range(min(n_t, kp2d.shape[1])):
        for k in range(kp2d.shape[2]):
            X = triangulate(kp2d[:, t, k, :2], kp2d[:, t, k, 2], Plist, prob_min)
            if not np.isfinite(X).all():
                continue
            res += [np.linalg.norm(reproject(X, P) - kp2d[c, t, k, :2])
                    for c, P in enumerate(Plist) if kp2d[c, t, k, 2] >= prob_min]
    return float(np.median(res)) if res else float("inf")


def solve_all(kp2d: np.ndarray, Ps: list[np.ndarray], prob_min: float):
    """Triangulate every (frame, joint): (T, K, 3) points and (T, K) mean residual."""
    T, K = kp2d.shape[1], kp2d.shape[2]
    kp3d = np.full((T, K, 3), np.nan)
    resid = np.full((T, K), np.nan)
    for t in range(T):
        for k in range(K):
            X = triangulate(kp2d[:, t, k, :2], kp2d[:, t, k, 2], Ps, prob_min)
            kp3d[t, k] = X
            if np.isfinite(X).all():
                errs = [np.linalg.norm(reproject(X, P) - kp2d[c, t, k, :2])
                        for c, P in enumerate(Ps) if kp2d[c, t, k, 2] >= prob_min]
                resid[t, k] = np.mean(errs) if errs else np.nan
    return kp3d, resid


def solve_camera_order(kp2d: np.ndarray, Ps: list[np.ndarray], prob_min: float):
    """Which P belongs to which video file?

    `camera-{i}.mp4` does not always line up with `cam_mat_all[:, :, i]` — on
    rec10-M2 the identity order gives 157 px median reprojection error while a
    cyclic shift gives 24 px. The mapping is not documented, so measure it:
    24 permutations, pick the one that reprojects best. Cheap and self-checking.
    """
    scored = sorted((median_residual(kp2d, [Ps[p[c]] for c in range(len(Ps))], prob_min), p)
                    for p in itertools.permutations(range(len(Ps))))
    best_r, best_p = scored[0]
    ident_r = median_residual(kp2d, Ps, prob_min)
    print(f"camera↔P order: {best_p} ({best_r:.1f} px) | identity {ident_r:.1f} px "
          f"| runner-up {scored[1][1]} ({scored[1][0]:.1f} px)")
    return list(best_p), best_r, ident_r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--snapshot", type=Path, default=Path(
        "/node_data_2/joon/data/external/SBeA/sbea_release_assets/"
        "fig2_data/well-trained models/dlc/snapshot-1030000"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=600)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--prob-min", type=float, default=0.6)
    ap.add_argument("--resolve-lr", action="store_true",
                    help="relabel bilateral joints so the four views agree "
                         "(partial fix — read sbea_lr_resolve's docstring first)")
    ap.add_argument("--lr-lambda", type=float, default=lr.DEFAULT_LAMBDA,
                    help="switch penalty for --resolve-lr; 0 flickers badly")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cma = np.array(sio.loadmat(
        str(a.root / f"{a.session}-caliParas.mat"))["caliParams"][0, 0]["cam_mat_all"])
    Ps = [cma[:, :, i].astype(np.float64) for i in range(N_CAMS)]

    import tensorflow.compat.v1 as tf
    n_joints = tf.train.load_checkpoint(
        str(a.snapshot)).get_variable_to_shape_map()["pose/part_pred/block4/biases"][0]
    print(f"snapshot → num_joints={n_joints}, joints={BODYPARTS[:n_joints]}")

    from deeplabcut.pose_estimation_tensorflow.core import predict
    cfg = build_pose_cfg(str(a.snapshot), n_joints)
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    frames = list(range(a.start, a.end, a.step))
    kp2d = np.stack([                                  # (cam, T, K, 3)
        infer_camera(a.root / f"{a.session}-camera-{c}.mp4", frames, cfg, sess, inputs, outputs)
        for c in range(N_CAMS)])
    print(f"2D done: {kp2d.shape}, mean likelihood {np.nanmean(kp2d[..., 2]):.3f}")

    order, best_res, ident_res = solve_camera_order(kp2d, Ps, a.prob_min)
    Ps = [Ps[i] for i in order]

    if a.resolve_lr:
        # print the independent check alongside, because the relabelling minimises
        # reprojection error and so cannot be judged by it — see sbea_lr_resolve
        before = solve_all(kp2d, Ps, a.prob_min)[0]
        kp2d, switches = lr.resolve(kp2d, Ps, a.prob_min, BODYPARTS[:n_joints],
                                    triangulate, reproject, lam=a.lr_lambda)
        after = solve_all(kp2d, Ps, a.prob_min)[0]
        print(f"L/R resolve (lam={a.lr_lambda:g}): {switches} pattern switches over "
              f"{len(frames)} frames | {lr.report(before, after, BODYPARTS[:n_joints])}")

    kp3d, resid = solve_all(kp2d, Ps, a.prob_min)
    K = n_joints
    valid = np.isfinite(kp3d).all(axis=2)
    print(f"3D  done: {kp3d.shape}, valid {valid.mean():.1%}, "
          f"reproj residual median {np.nanmedian(resid):.2f} px")

    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"{a.session}-kp3d.npz"
    np.savez_compressed(
        dst, keypoints_3d=kp3d, confidence=np.nanmin(kp2d[..., 2], axis=0),
        reproj_residual_px=resid, frame_indices=np.array(frames),
        keypoint_names=np.array(BODYPARTS[:K]), keypoints_2d=kp2d, prob_min=a.prob_min,
        camera_order=np.array(order), P_matrices=np.stack(Ps),
        residual_identity_px=ident_res, residual_best_px=best_res,
        lr_resolved=a.resolve_lr)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
