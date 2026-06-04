"""1D Kalman filter + RTS smoother per (keypoint, axis) for 3D KP series.

Theory
------
State (2D): x_k = [p_k, v_k]^T (position, velocity).
Transition (constant velocity): x_k = F · x_{k-1} + w_k
  F = [[1, dt], [0, 1]],  process noise Q = q · [[dt^4/4, dt^3/2],
                                                  [dt^3/2, dt^2  ]]
Observation: z_k = H · x_k + v_k,  H = [1, 0], R = measurement variance.

Forward Kalman pass: posterior x_post[k], P_post[k], plus prior x_pred[k+1].
RTS backward smoother: x_smooth[k] = x_post[k] + C · (x_smooth[k+1] - x_pred[k+1])
                       C = P_post[k] · F^T · P_pred[k+1]^{-1}

NaN handling: prediction-only (skip update) at NaN measurements.

Hyperparameters
---------------
q (process noise scale): velocity volatility (mm^2 / frame^4).
                          Higher q → less smoothing.
R (measurement noise variance): per-frame triangulation jitter (mm^2).
                                Lower R → trust measurement more.

For our data (~100 fps mouse, triangulation noise ~2 mm):
  q = 1.0    (allow velocity changes ~1 mm/frame^2)
  R = 4.0    (~2 mm std)

Reference: Rauch, Tung, Striebel (1965) "Maximum likelihood estimates
of linear dynamic systems."
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED = REPO_ROOT / "outputs/kp_benchmark"


def kalman_rts_1d(z: np.ndarray, q: float = 1.0, R: float = 4.0,
                   dt: float = 1.0) -> np.ndarray:
    """Constant-velocity Kalman + RTS smoother on a 1D sequence.

    z : (T,) measurements, NaN allowed (skip update).
    Returns smoothed (T,) positions.
    """
    T = len(z)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt**4 / 4, dt**3 / 2],
                       [dt**3 / 2, dt**2     ]])
    I = np.eye(2)

    # Find first valid measurement for initialization
    finite = np.where(np.isfinite(z))[0]
    if len(finite) == 0:
        return z.copy()
    k0 = finite[0]
    x = np.zeros(2)
    x[0] = z[k0]
    P = np.diag([R, q])

    # Storage for forward pass
    x_post = np.full((T, 2), np.nan)
    P_post = np.full((T, 2, 2), np.nan)
    x_pred = np.full((T, 2), np.nan)
    P_pred = np.full((T, 2, 2), np.nan)

    x_post[k0] = x
    P_post[k0] = P

    # Forward pass
    for k in range(k0 + 1, T):
        # Predict
        x_p = F @ x_post[k - 1]
        P_p = F @ P_post[k - 1] @ F.T + Q
        x_pred[k] = x_p
        P_pred[k] = P_p

        if np.isfinite(z[k]):
            # Update
            y = z[k] - (H @ x_p)[0]            # innovation
            S = (H @ P_p @ H.T + R)[0, 0]      # innovation cov (scalar)
            K = (P_p @ H.T / S).flatten()      # 2-vector
            x_post[k] = x_p + K * y
            P_post[k] = (I - np.outer(K, H[0])) @ P_p
        else:
            x_post[k] = x_p
            P_post[k] = P_p

    # Backward RTS smoother
    x_smooth = x_post.copy()
    P_smooth = P_post.copy()
    for k in range(T - 2, k0 - 1, -1):
        if not np.isfinite(P_pred[k + 1, 0, 0]):
            continue
        try:
            P_pred_inv = np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        C = P_post[k] @ F.T @ P_pred_inv
        x_smooth[k] = x_post[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_post[k] + C @ (P_smooth[k + 1] - P_pred[k + 1]) @ C.T

    # Restore NaN where original was NaN (to be honest about missing data)
    out = x_smooth[:, 0].copy()
    out[~np.isfinite(z)] = np.nan
    return out


def kalman_rts_kp(kp: np.ndarray, q: float = 1.0, R: float = 4.0) -> np.ndarray:
    """Apply 1D Kalman+RTS to (T, K, 3) keypoint sequence per (k, axis)."""
    out = np.empty_like(kp, dtype=np.float64)
    T, K, D = kp.shape
    for k in range(K):
        for d in range(D):
            out[:, k, d] = kalman_rts_1d(kp[:, k, d].astype(np.float64), q=q, R=R)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["dlc_resnet50_imagenet"])
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--R", type=float, default=4.0)
    args = ap.parse_args()

    for tag in args.tags:
        src = PRED / f"{tag}_full_kp.npz"
        if not src.exists():
            print(f"skip: {src}"); continue
        d = np.load(src)
        kp = d["keypoints_3d"].astype(np.float64)
        print(f"\n[{tag}] q={args.q} R={args.R}")
        before_vel = np.nanmean(np.linalg.norm(np.diff(kp, axis=0), axis=-1))
        sm = kalman_rts_kp(kp, q=args.q, R=args.R)
        after_vel = np.nanmean(np.linalg.norm(np.diff(sm, axis=0), axis=-1))
        print(f"  vel before {before_vel:.3f} → after {after_vel:.3f} mm/fr "
              f"({(1 - after_vel / before_vel) * 100:.1f}% jitter reduction)")
        out = PRED / f"{tag}_full_kp_kalman.npz"
        # Match smooth_kp_temporal.py schema: include valid_mask so downstream
        # scripts can use either smoother interchangeably (review I2).
        valid_mask = ~np.isnan(sm).any(axis=-1)
        np.savez(out,
                 keypoints_3d=sm.astype(np.float32),
                 valid_mask=valid_mask,
                 frame_ids=d["frame_ids"],
                 keypoint_names=d["keypoint_names"],
                 q=args.q, R=args.R)
        print(f"  saved → {out.name}")


if __name__ == "__main__":
    main()
