"""Figure and overlay production for the SBeA verification report.

Split out of `render_sbea_report.py` at 260817 (that file crossed the 400-line cap).
Boundary: this module turns artefacts into base64 images and HTML fragments; the caller
decides what goes in the report. Nothing here reads argv or writes files.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# same correction applied to KP22 (260702), not a single mid-body hub.
EDGES = [(0, 1), (0, 2), (0, 3), (3, 12), (12, 13), (13, 14), (14, 15),
         (3, 4), (4, 8), (3, 5), (5, 9), (13, 6), (6, 10), (13, 7), (7, 11)]

# Region palette carried over from BehaviorSplatter so the two projects' overlays read the
# same (SSOT = BehaviorSplatter/src/behaviorsplatter/visualization/keypoints.py::KP_COLORS).
# Left and right get different colours on purpose: an L/R swap — the dominant residual
# component here — then shows up as a wrong-coloured limb instead of hiding in the numbers.
_HEAD, _BODY, _TAIL = (255, 255, 0), (255, 0, 255), (255, 165, 0)
_LF, _RF, _LH, _RH = (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0)
REGION = {0: _HEAD, 1: _HEAD, 2: _HEAD, 3: _BODY, 12: _BODY,
          13: _TAIL, 14: _TAIL, 15: _TAIL,
          4: _LF, 8: _LF, 5: _RF, 9: _RF, 6: _LH, 10: _LH, 7: _RH, 11: _RH}
REGION_LEGEND = [("머리", _HEAD), ("몸통", _BODY), ("꼬리", _TAIL), ("좌앞", _LF),
                 ("우앞", _RF), ("좌뒤", _LH), ("우뒤", _RH)]


def bgr(i: int) -> tuple[int, int, int]:
    """KP_COLORS is RGB; cv2 draws BGR."""
    r, g, b = REGION.get(i, (200, 200, 200))
    return (b, g, r)


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_b64(img: np.ndarray, width: int = 460) -> str:
    s = width / img.shape[1]
    small = cv2.resize(img, (width, max(1, int(img.shape[0] * s))))
    return base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 86])[1]).decode()


def overlay_2d(video: Path, frame: int, kp: np.ndarray, thr: float) -> str | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    for a, b in EDGES:
        if kp[a, 2] >= thr and kp[b, 2] >= thr:
            cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), bgr(b), 2)
    for i, (x, y, q) in enumerate(kp):
        if np.isfinite([x, y]).all():
            cv2.circle(img, (int(x), int(y)), 4, bgr(i), -1)
            if q < thr:                                   # below threshold = hollow, not recoloured
                cv2.circle(img, (int(x), int(y)), 6, (255, 255, 255), 1)
    return img_b64(img)


def read_mask(mask_npz: Path) -> np.ndarray | None:
    """Union of the animal masks in one npz.

    Three key conventions are in circulation and a consumer must read all three
    (sdannce-poc/docs/repo_boundary.md, 260727): `animal_N` from the current kp-SAM2
    producer, bare `mask` from BS `auto_sam3`, and legacy `ratN`. Reading only
    `animal_*` silently returns nothing for the sam3 tree.
    """
    if not mask_npz.exists():
        return None
    with np.load(mask_npz) as z:
        keys = [k for k in z.files if k.startswith(("animal_", "rat"))] or \
               [k for k in z.files if k == "mask"]
        if not keys:
            return None
        return np.logical_or.reduce([z[k].astype(bool) for k in keys])


def cam_dir(root: Path, c: int) -> Path | None:
    """Per-camera mask dir. kp-SAM2 writes `Camera1..4`, auto_sam3 writes `cam0..3`."""
    for name in (f"Camera{c + 1}", f"cam{c}"):
        if (root / name).is_dir():
            return root / name
    return None


def overlay_mask(video: Path, frame: int, mask_npz: Path) -> str | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    m = read_mask(mask_npz) if ok else None
    if not ok or m is None:
        return None
    img[m] = (0.5 * np.array([255, 120, 0]) + 0.5 * img[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (255, 120, 0), 2)
    return img_b64(img)


def overlay_reproj(video: Path, frame: int, det: np.ndarray, proj: np.ndarray,
                   thr: float) -> str | None:
    """Detected 2D (green) vs the 3D solution projected back (red), joined by a line.

    This is the check the residual number stands for: if the red dots sit on the green
    ones the triangulation agrees with the detector, and the line length IS the residual.
    Overlaying only the detections (as this report did before) cannot show that.
    """
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    for (x, y, q), (u, v) in zip(det, proj):
        if not (np.isfinite([x, y, u, v]).all() and q >= thr):
            continue
        cv2.line(img, (int(x), int(y)), (int(u), int(v)), (200, 200, 200), 1)
        cv2.circle(img, (int(x), int(y)), 4, (0, 220, 0), -1)
        cv2.circle(img, (int(u), int(v)), 4, (0, 60, 235), -1)
    return img_b64(img)


def _tri(xy: np.ndarray, mask: np.ndarray, Ps: np.ndarray) -> np.ndarray:
    """Linear DLT for one frame — same solver the pipeline uses, imported to avoid a second copy."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sbea_bundle_adjust as ba
    return ba.triangulate_batch(xy, mask, np.asarray(Ps))


def project(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    h = np.concatenate([X, np.ones((len(X), 1))], 1) @ P.T
    w = h[:, 2:]
    return np.where(np.abs(w) > 1e-9, h[:, :2] / np.where(w == 0, 1, w), np.nan)


def fig_residual(resid: np.ndarray, names: list[str]) -> tuple[str, str]:
    per_kp = np.nanmedian(resid, axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(range(len(names)), per_kp, color="#4c8bf5")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("median reproj. error (px)")
    ax.grid(axis="y", alpha=.25)
    bar = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    v = resid[np.isfinite(resid)]
    ax.hist(v, bins=50, color="#4c8bf5", alpha=.85)
    ax.axvline(np.median(v), color="#e2574c", ls="--", label=f"median {np.median(v):.1f} px")
    ax.set_xlabel("reproj. error (px)")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=.25)
    return bar, fig_b64(fig)


def mismatch_rows(resid: np.ndarray, names: list[str]) -> tuple[str, int]:
    """Per-joint residual vs the midline floor, with the L/R partner's value beside it.

    Two flags, both quantitative rather than eyeballed:
      OUTLIER  — joint median > 1.5x the midline median (midline = no L/R ambiguity possible)
      L/R GAP  — a bilateral pair differs by > 30%, the signature of a per-camera label swap.
    """
    med = np.nanmedian(resid, axis=0)
    mid = [i for i, n in enumerate(names) if not n.startswith(("left_", "right_"))]
    floor = float(np.nanmedian(med[mid]))
    idx = {n: i for i, n in enumerate(names)}
    rows, n_flag = "", 0
    for i, n in enumerate(names):
        partner = idx.get(("right_" + n[5:]) if n.startswith("left_") else
                          ("left_" + n[6:]) if n.startswith("right_") else "")
        flags = []
        if med[i] > 1.5 * floor:
            flags.append("OUTLIER")
        if partner is not None and max(med[i], med[partner]) > 1.3 * min(med[i], med[partner]):
            flags.append("L/R GAP")
        n_flag += bool(flags)
        c = "#e2574c" if flags else "#666"
        rows += (f"<tr><td>{n}</td><td class='k'>{med[i]:.1f}</td>"
                 f"<td class='k'>{med[i] / floor:.2f}×</td>"
                 f"<td class='k'>{'' if partner is None else f'{med[partner]:.1f}'}</td>"
                 f"<td style='color:{c}'>{' · '.join(flags) or '—'}</td></tr>")
    return rows, n_flag


def fig_skeleton(kp3d: np.ndarray, frames: list[int]) -> str:
    fig = plt.figure(figsize=(9, 3.2))
    for i, t in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), i + 1, projection="3d")
        P = kp3d[t]
        for a, b in EDGES:
            if np.isfinite(P[[a, b]]).all():
                ax.plot(*zip(P[a], P[b]), c="#4c8bf5", lw=1.4)
        ok = np.isfinite(P).all(axis=1)
        ax.scatter(*P[ok].T, c="#e2574c", s=9)
        ax.set_title(f"t={t}", fontsize=9)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    return fig_b64(fig)


def fig_mask_area(masks: Path, cams: list[int], max_pts: int = 400) -> str:
    """Mask area over time — a flat-ish line means the segmenter kept tracking the animal."""
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for c in cams:
        d = cam_dir(masks, c)
        files = sorted(d.glob("mask_*.npz")) if d else []
        files = files[:: max(1, len(files) // max_pts)]      # sam3 tree has ~8k/cam
        xs, ys = [], []
        for f in files:
            m = read_mask(f)
            if m is not None:
                xs.append(int(f.stem.split("_")[1]))
                ys.append(int(m.sum()))
        if xs:
            ax.plot(xs, ys, lw=1.3, label=f"camera-{c}")
    ax.set_xlabel("frame"); ax.set_ylabel("mask area (px)")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    return fig_b64(fig)


