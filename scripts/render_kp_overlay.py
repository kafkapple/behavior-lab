"""Render real-frame KP overlay visualization for kp_benchmark v0.1.

For a chosen sample video frame:
  1. Read frame from all 6 cameras of M1.
  2. Project MAMMAL 22-kp 3D → 2D using calibrated camera params from
     label3d_dannce.mat (K, R, t + radial/tangential distortion).
  3. Overlay projected points + skeleton on the frame.
  4. If the frame is in Li GT set, overlay Li GT keypoints in a second color.
  5. Save a 6-view grid PNG.

Also serves as a coordinate-system sanity check — if projected MAMMAL keypoints
don't land on the mouse body, the (K, R, t) interpretation is wrong.

Usage
-----
    python scripts/render_kp_overlay.py \\
        [--video-dir DIR] [--frame-idx N] [--output PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

KP_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end",
    "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip",
    "R_foot", "R_knee", "R_hip",
]

EDGES = [
    (2, 3), (0, 3), (1, 3),
    (3, 4), (4, 5), (5, 6), (6, 7),
    (4, 11), (4, 15),
    (11, 10), (10, 8), (8, 9),
    (15, 14), (14, 12), (12, 13),
    (5, 18), (5, 21),
    (18, 17), (17, 16),
    (21, 20), (20, 19),
]

COLOR_MAMMAL = (0, 200, 0)        # green BGR
COLOR_MAMMAL_BONE = (0, 140, 0)
COLOR_LI = (40, 40, 230)          # red-ish BGR
COLOR_LI_BONE = (20, 20, 180)


def load_cameras(label3d_path: Path) -> list[dict]:
    """Parse 6 camera params from label3d_dannce.mat."""
    mat = sio.loadmat(label3d_path, struct_as_record=False, squeeze_me=True)
    cams = []
    for i in range(len(mat["camnames"])):
        p = mat["params"][i]
        p = p.item() if hasattr(p, "item") else p
        K = np.asarray(p.K, dtype=np.float64).T
        R = np.asarray(p.r, dtype=np.float64).T
        t = np.asarray(p.t, dtype=np.float64).reshape(3)
        if np.linalg.det(R) < 0:
            R = R.copy()
            R[:, 2] *= -1
        cams.append({
            "name": str(mat["camnames"][i]),
            "K": K, "R": R, "t": t,
            "rdist": np.asarray(p.RDistort, dtype=np.float64).flatten(),
            "tdist": np.asarray(p.TDistort, dtype=np.float64).flatten(),
        })
    return cams


def project(pts_3d: np.ndarray, cam: dict, distort: bool = False) -> np.ndarray:
    """Project (N, 3) world points → (N, 2) pixel coords.

    distort=False (default): pinhole only — use for videos_undist (already
    undistorted by upstream pipeline). Applying distortion to projections
    onto undistorted images causes systematic edge-of-frame offsets.

    distort=True: apply radial (k1, k2, k3) + tangential (p1, p2) distortion
    from cam params. Use only when projecting onto raw distorted images.
    """
    K, R, t = cam["K"], cam["R"], cam["t"]

    cam_pts = (R @ pts_3d.T).T + t                  # (N, 3) camera frame
    z = cam_pts[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = cam_pts[:, 0] / safe
    y = cam_pts[:, 1] / safe

    if distort:
        rdist = cam["rdist"]
        tdist = cam["tdist"]
        r2 = x * x + y * y
        k1 = rdist[0] if rdist.size > 0 else 0.0
        k2 = rdist[1] if rdist.size > 1 else 0.0
        k3 = rdist[2] if rdist.size > 2 else 0.0
        p1 = tdist[0] if tdist.size > 0 else 0.0
        p2 = tdist[1] if tdist.size > 1 else 0.0
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x + cx
    v = fy * y + cy
    pixels = np.stack([u, v], axis=-1)
    pixels[z <= 0] = np.nan
    return pixels


def draw_kp(img: np.ndarray, pts_2d: np.ndarray, color_pt, color_bone,
            radius: int = 4, label: str | None = None) -> None:
    h, w = img.shape[:2]
    for a, b in EDGES:
        pa, pb = pts_2d[a], pts_2d[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     color_bone, 2, cv2.LINE_AA)
    for i, p in enumerate(pts_2d):
        if not np.all(np.isfinite(p)):
            continue
        u, v = int(p[0]), int(p[1])
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img, (u, v), radius, color_pt, -1, cv2.LINE_AA)
    if label is not None:
        cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color_pt, 2, cv2.LINE_AA)


def render_frame_grid(
    video_dir: Path,
    cams: list[dict],
    frame_idx: int,
    mammal_kp_3d: np.ndarray | None,
    li_kp_3d: np.ndarray | None,
    li_valid_mask: np.ndarray | None,
) -> np.ndarray:
    tiles = []
    for cam_i, cam in enumerate(cams):
        vp = video_dir / f"{cam_i}.mp4"
        cap = cv2.VideoCapture(str(vp))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = np.zeros((1024, 1152, 3), dtype=np.uint8)
            cv2.putText(frame, f"missing frame {frame_idx} in cam {cam_i}",
                        (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        if mammal_kp_3d is not None:
            pts2d = project(mammal_kp_3d, cam)
            draw_kp(frame, pts2d, COLOR_MAMMAL, COLOR_MAMMAL_BONE,
                    radius=5, label=f"{cam['name']} | MAMMAL (green)")
        if li_kp_3d is not None:
            li_pts2d = project(li_kp_3d, cam)
            if li_valid_mask is not None:
                li_pts2d[~li_valid_mask] = np.nan
            draw_kp(frame, li_pts2d, COLOR_LI, COLOR_LI_BONE,
                    radius=4, label=None)
            cv2.putText(frame, "Li GT (red)", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LI, 2, cv2.LINE_AA)

        cv2.putText(frame, f"video frame {frame_idx}",
                    (frame.shape[1] - 280, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(frame)

    # 2 × 3 grid
    h_tile = 512
    resized = []
    for t in tiles:
        scale = h_tile / t.shape[0]
        resized.append(cv2.resize(t, (int(t.shape[1] * scale), h_tile)))
    w_tile = min(im.shape[1] for im in resized)
    resized = [im[:, :w_tile] for im in resized]
    row1 = np.hstack(resized[:3])
    row2 = np.hstack(resized[3:])
    return np.vstack([row1, row2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video-dir", type=Path,
        default=Path.home() / "data/external/MAMMAL_Mesh_markerless_mouse_1"
                              "/data/markerless_mouse_1_nerf/videos_undist",
    )
    ap.add_argument(
        "--label3d", type=Path,
        default=REPO_ROOT / "data/markerless_mouse_1/labels/label3d_dannce.mat",
    )
    ap.add_argument(
        "--mammal-npz", type=Path,
        default=REPO_ROOT / "data/mammal_mouse/v012345_kp22_20260126"
                            "/keypoints_22_3d.npz",
    )
    ap.add_argument(
        "--li-gt-npz", type=Path,
        default=REPO_ROOT / "data/markerless_mouse_1/labels/li_m1_gt.npz",
    )
    ap.add_argument("--frames", type=int, nargs="+",
                    default=None,
                    help="video frame indices; default = pick MAMMAL+Li overlap samples")
    ap.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs/kp_benchmark/overlay",
    )
    args = ap.parse_args()

    cams = load_cameras(args.label3d)
    print(f"[overlay] loaded {len(cams)} cameras from {args.label3d.name}")

    mam = np.load(args.mammal_npz)
    mammal_kp = mam["keypoints"].astype(np.float64)
    mammal_idx = mam["frame_indices"]

    li = np.load(args.li_gt_npz)
    li_kp = li["keypoints_3d"].astype(np.float64)
    li_fi = li["frame_ids"]
    li_valid = li["valid_mask"]

    # Pick frames: prefer overlap of MAMMAL grid and Li GT
    if args.frames is None:
        overlap = np.intersect1d(mammal_idx, li_fi)
        if len(overlap) >= 2:
            args.frames = [int(overlap[0]), int(overlap[len(overlap) // 2])]
        else:
            args.frames = [int(mammal_idx[len(mammal_idx) // 2])]
        print(f"[overlay] auto-selected frames: {args.frames}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for frame_idx in args.frames:
        mammal_kp_for_frame = None
        if frame_idx in mammal_idx:
            i = int(np.where(mammal_idx == frame_idx)[0][0])
            mammal_kp_for_frame = mammal_kp[i]

        li_kp_for_frame = None
        li_valid_for_frame = None
        if frame_idx in li_fi:
            j = int(np.where(li_fi == frame_idx)[0][0])
            li_kp_for_frame = li_kp[j]
            li_valid_for_frame = li_valid[j]

        grid = render_frame_grid(
            args.video_dir, cams, frame_idx,
            mammal_kp_for_frame, li_kp_for_frame, li_valid_for_frame,
        )
        out = args.output_dir / f"frame_{frame_idx:05d}_overlay.png"
        cv2.imwrite(str(out), grid)
        saved.append(out)
        print(f"  saved {out.name} ({out.stat().st_size / 1024:.1f} KB) "
              f"shape={grid.shape}")

    print(f"[done] {len(saved)} overlay grid(s) → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
