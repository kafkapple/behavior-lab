"""Render per-model prediction overlay videos (6-cam grid, full 18000 frames).

Outputs 2 MP4 files:
  rn50_predictions_grid.mp4    — 6-cam grid + RN50 cyan keypoint overlay
  sa_predictions_grid.mp4      — 6-cam grid + SA zero-shot orange overlay

Each video is 18000 frames @ 100 fps = 3 minutes playback. Reads videos
sequentially (no seek) for speed. Grid resolution downscaled 2× to keep
file size manageable.

Usage:
    python scripts/render_prediction_videos.py [--subsample 1]
        --subsample N → keep every N-th frame (1 = full 18000)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_kp_overlay import load_cameras, project, EDGES, KP_NAMES

VIDEO_DIR = Path.home() / "data/external/MAMMAL_Mesh_markerless_mouse_1/data/markerless_mouse_1_nerf/videos_undist"
DATA = REPO_ROOT / "data"
PRED_DIR = REPO_ROOT / "outputs/kp_benchmark"
OUT_DIR = Path(
    "/Users/joon/Documents/Obsidian/30_Projects/"
    "2603_3D_animal_recon_BehaviorSplatter/_html"
)


def draw_overlay(img, pts2d, color_pt, color_bone, radius=4):
    h, w = img.shape[:2]
    for a, b in EDGES:
        pa, pb = pts2d[a], pts2d[b]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        try:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     color_bone, 2, cv2.LINE_AA)
        except (cv2.error, ValueError, OverflowError):
            pass
    for p in pts2d:
        if not np.all(np.isfinite(p)):
            continue
        try:
            u, v = int(p[0]), int(p[1])
        except (ValueError, OverflowError):
            continue
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img, (u, v), radius, color_pt, -1, cv2.LINE_AA)


def render_video(tag, kp_3d, cams, color_pt, color_bone,
                 out_path, subsample=1, tile_h=384):
    """Render 2×3 grid video with overlay for a single model.

    kp_3d: (N, 22, 3) — predictions
    """
    N = kp_3d.shape[0]
    # Open all 6 video files
    caps = [cv2.VideoCapture(str(VIDEO_DIR / f"{i}.mp4")) for i in range(6)]
    src_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = tile_h / src_h
    tile_w = int(src_w * scale)
    grid_w = tile_w * 3
    grid_h = tile_h * 2

    fps_in = caps[0].get(cv2.CAP_PROP_FPS)
    fps_out = fps_in / subsample
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed: {out_path}")

    print(f"[{tag}] {N} src frames → subsample={subsample} → "
          f"{N // subsample} output frames @ {fps_out:.1f} fps, "
          f"grid {grid_w}×{grid_h}")

    t0 = time.time()
    written = 0
    label_color = (255, 255, 255)
    for fi in range(N):
        if fi % subsample != 0:
            # Still need to advance reads — read & discard
            for cap in caps:
                cap.grab()
            continue
        tiles = []
        for cam_i, cap in enumerate(caps):
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((src_h, src_w, 3), dtype=np.uint8)
            kp = kp_3d[fi]
            if np.any(np.isfinite(kp)):
                pts2d = project(kp.astype(np.float64), cams[cam_i])
                draw_overlay(frame, pts2d, color_pt, color_bone, radius=5)
            cv2.putText(frame, f"cam{cam_i} | frame {fi}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, label_color, 2, cv2.LINE_AA)
            tile = cv2.resize(frame, (tile_w, tile_h))
            tiles.append(tile)
        # 2×3 grid
        row1 = np.hstack(tiles[:3])
        row2 = np.hstack(tiles[3:])
        grid = np.vstack([row1, row2])
        # Banner
        cv2.putText(grid, tag, (12, grid.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_pt, 2, cv2.LINE_AA)
        writer.write(grid)
        written += 1

        if fi % 600 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(written, 1) * (N // subsample - written)
            print(f"  [{tag}] frame {fi}/{N}, written {written}, "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    for cap in caps:
        cap.release()
    writer.release()
    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"[{tag}] done → {out_path} ({sz:.1f} MB, {written} frames, "
          f"total {time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsample", type=int, default=1,
                    help="keep every N-th frame (1=full 18000)")
    ap.add_argument("--tile-h", type=int, default=384,
                    help="per-cam tile height (pixels)")
    args = ap.parse_args()

    cams = load_cameras(DATA / "markerless_mouse_1/labels/label3d_dannce.mat")
    rn = np.load(PRED_DIR / "dlc_resnet50_imagenet_full_kp.npz")["keypoints_3d"]
    sa = np.load(PRED_DIR / "dlc_superanimal_zeroshot_hrnet_w32_full_kp.npz")["keypoints_3d"]

    # BGR colors
    CYAN_PT = (255, 200, 0); CYAN_BONE = (200, 150, 0)
    ORANGE_PT = (0, 140, 255); ORANGE_BONE = (0, 100, 200)

    render_video("DLC_ResNet50_trained", rn, cams,
                 CYAN_PT, CYAN_BONE,
                 OUT_DIR / "260603_kp_rn50_predictions_grid.mp4",
                 subsample=args.subsample, tile_h=args.tile_h)
    render_video("DLC_SuperAnimal_zeroshot", sa, cams,
                 ORANGE_PT, ORANGE_BONE,
                 OUT_DIR / "260603_kp_sa_zeroshot_predictions_grid.mp4",
                 subsample=args.subsample, tile_h=args.tile_h)


if __name__ == "__main__":
    main()
