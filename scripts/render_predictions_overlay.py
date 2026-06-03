"""Render real-frame overlay with BOTH model predictions + GT for kp_benchmark.

Adds RN50 (cyan) and SA zero-shot (orange) projected 2D points on top of
the existing MAMMAL pseudo-GT (green) + Li human GT (red) overlay. Saves
PNGs into outputs/kp_benchmark/overlay/.

Plus:
- 3D trajectory plot for body_middle (root joint) over 18000 frames
  comparing the two models.
- Per-frame MPJPE histogram comparing RN50 vs SA zero-shot.
"""
from __future__ import annotations

import io
import base64
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from render_kp_overlay import (load_cameras, project, draw_kp, render_frame_grid,
                                KP_NAMES, EDGES)

VIDEO_DIR = Path.home() / "data/external/MAMMAL_Mesh_markerless_mouse_1/data/markerless_mouse_1_nerf/videos_undist"
DATA = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs/kp_benchmark/overlay"
PRED_DIR = REPO_ROOT / "outputs/kp_benchmark"

COLOR_RN50 = (255, 200, 0)        # cyan-ish (BGR)
COLOR_RN50_BONE = (200, 150, 0)
COLOR_SA = (0, 140, 255)          # orange (BGR)
COLOR_SA_BONE = (0, 100, 200)


def render_four_way_grid(cams, frame_idx, mammal_3d, li_3d, li_valid,
                         rn_3d, sa_3d):
    """Like render_frame_grid but adds RN50 + SA predictions."""
    tiles = []
    for cam_i, cam in enumerate(cams):
        vp = VIDEO_DIR / f"{cam_i}.mp4"
        cap = cv2.VideoCapture(str(vp))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = np.zeros((1024, 1152, 3), dtype=np.uint8)

        if mammal_3d is not None:
            pts = project(mammal_3d, cam)
            draw_kp(frame, pts, (0, 200, 0), (0, 140, 0), radius=5,
                    label=f"cam{cam_i} | green=MAMMAL")
        if li_3d is not None:
            pts = project(li_3d, cam)
            if li_valid is not None:
                pts[~li_valid] = np.nan
            draw_kp(frame, pts, (40, 40, 230), (20, 20, 180), radius=4)
            cv2.putText(frame, "red=Li GT", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 230), 2, cv2.LINE_AA)
        if rn_3d is not None:
            pts = project(rn_3d, cam)
            draw_kp(frame, pts, COLOR_RN50, COLOR_RN50_BONE, radius=4)
            cv2.putText(frame, "cyan=RN50", (12, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RN50, 2, cv2.LINE_AA)
        if sa_3d is not None:
            mask = np.isfinite(sa_3d).all(axis=-1)
            if mask.any():
                pts = project(sa_3d, cam)
                pts[~mask] = np.nan
                draw_kp(frame, pts, COLOR_SA, COLOR_SA_BONE, radius=4)
                cv2.putText(frame, "orange=SA zero-shot", (12, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SA, 2, cv2.LINE_AA)

        cv2.putText(frame, f"video frame {frame_idx}",
                    (frame.shape[1] - 280, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(frame)

    h_tile = 512
    resized = []
    for t in tiles:
        scale = h_tile / t.shape[0]
        resized.append(cv2.resize(t, (int(t.shape[1] * scale), h_tile)))
    w_tile = min(im.shape[1] for im in resized)
    resized = [im[:, :w_tile] for im in resized]
    return np.vstack([np.hstack(resized[:3]), np.hstack(resized[3:])])


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    cams = load_cameras(DATA / "markerless_mouse_1/labels/label3d_dannce.mat")
    mam = np.load(DATA / "mammal_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
    mam_kp = mam["keypoints"].astype(np.float64); mam_idx = mam["frame_indices"]
    li = np.load(DATA / "markerless_mouse_1/labels/li_m1_gt.npz")
    li_kp = li["keypoints_3d"].astype(np.float64); li_ids = li["frame_ids"]
    li_valid = li["valid_mask"]
    rn = np.load(PRED_DIR / "dlc_resnet50_imagenet_full_kp.npz")["keypoints_3d"].astype(np.float64)
    sa = np.load(PRED_DIR / "dlc_superanimal_zeroshot_hrnet_w32_full_kp.npz")["keypoints_3d"].astype(np.float64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pick 3 representative frames
    overlap = np.intersect1d(mam_idx, li_ids)
    sample_frames = [int(overlap[0]), int(overlap[len(overlap) // 2]),
                     int(mam_idx[len(mam_idx) // 2])]
    print(f"sample frames: {sample_frames}")

    for fi in sample_frames:
        m3 = mam_kp[np.where(mam_idx == fi)[0][0]] if fi in mam_idx else None
        l3 = li_kp[np.where(li_ids == fi)[0][0]] if fi in li_ids else None
        lv = li_valid[np.where(li_ids == fi)[0][0]] if fi in li_ids else None
        r3 = rn[fi] if fi < len(rn) else None
        s3 = sa[fi] if fi < len(sa) else None
        grid = render_four_way_grid(cams, fi, m3, l3, lv, r3, s3)
        out = OUT_DIR / f"frame_{fi:05d}_predictions.png"
        cv2.imwrite(str(out), grid)
        print(f"  saved {out.name} ({out.stat().st_size / 1024:.1f} KB)")

    # 3D trajectory of body_middle (idx 4) over 18000 frames
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d")
    valid_rn = np.isfinite(rn[:, 4, :]).all(axis=-1)
    valid_sa = np.isfinite(sa[:, 4, :]).all(axis=-1)
    rn_t = rn[valid_rn, 4, :]
    sa_t = sa[valid_sa, 4, :]
    step = 30
    ax.plot(rn_t[::step, 0], rn_t[::step, 1], rn_t[::step, 2],
            "-", color="#06a", alpha=0.7, lw=0.6, label=f"RN50 (n={valid_rn.sum()})")
    ax.plot(sa_t[::step, 0], sa_t[::step, 1], sa_t[::step, 2],
            "-", color="#e80", alpha=0.7, lw=0.6, label=f"SA zero-shot (n={valid_sa.sum()})")
    ax.set_title("body_middle 3D trajectory (subsampled 1/30)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.legend()

    ax2 = fig.add_subplot(122)
    rn_z = rn[:, 4, 2]
    sa_z = sa[:, 4, 2]
    ax2.plot(np.arange(18000), rn_z, color="#06a", alpha=0.6, lw=0.4, label="RN50 z")
    ax2.plot(np.arange(18000), sa_z, color="#e80", alpha=0.6, lw=0.4, label="SA zero-shot z")
    ax2.set_xlabel("video frame index")
    ax2.set_ylabel("body_middle Z (mm)")
    ax2.set_title("body_middle Z over 18000 frames")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    traj_path = OUT_DIR / "trajectory_body_middle.png"
    fig.savefig(traj_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {traj_path.name}")

    # Per-frame MPJPE histogram (vs MAMMAL pseudo-GT on 5-step grid)
    rn_at_mam = rn[mam_idx]
    sa_at_mam = sa[mam_idx]
    err_rn = np.linalg.norm((rn_at_mam - rn_at_mam[:, 4:5]) - (mam_kp - mam_kp[:, 4:5]), axis=-1)
    err_sa = np.linalg.norm((sa_at_mam - sa_at_mam[:, 4:5]) - (mam_kp - mam_kp[:, 4:5]), axis=-1)
    pf_rn = np.nanmean(err_rn, axis=-1)
    pf_sa = np.nanmean(err_sa, axis=-1)

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.linspace(0, max(np.nanmax(pf_rn), np.nanmax(pf_sa)) + 5, 50)
    ax.hist(pf_rn[np.isfinite(pf_rn)], bins=bins, alpha=0.6, color="#06a", label=f"RN50 mean={np.nanmean(pf_rn):.1f} mm")
    ax.hist(pf_sa[np.isfinite(pf_sa)], bins=bins, alpha=0.6, color="#e80", label=f"SA zero-shot mean={np.nanmean(pf_sa):.1f} mm")
    ax.set_xlabel("Per-frame root-relative MPJPE (mm)")
    ax.set_ylabel("Frame count")
    ax.set_title("Per-frame MPJPE distribution (vs MAMMAL pseudo-GT, 3600 frames)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    hist_path = OUT_DIR / "per_frame_mpjpe_hist.png"
    fig.savefig(hist_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {hist_path.name}")

    print(f"[done] all visuals → {OUT_DIR}")


if __name__ == "__main__":
    main()
