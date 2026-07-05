"""Dyadic ego-centric feature extraction for 2-mouse social behavior.

The standard practice for CalMS21 / MARS / MABe-style social behavior
analysis is **not** to feed raw pixel coordinates of one animal — both
mice and their *relation* matter (Segalin et al. 2021 MARS;
Sun et al. 2021 CalMS21; Sun et al. 2021 TREBA).

This module implements the canonical ego-centric transform:

    1. translate so resident's centroid sits at the origin
    2. rotate so resident's body axis (tail_base → neck) aligns with +y
    3. emit (a) resident keypoints in this frame, (b) intruder keypoints
       in this frame, (c) inter-animal scalar/angular descriptors

Degeneracy guards (MoA Layer 2 consensus):
    * body axis norm < EPS → fall back to last valid rotation
    * angles encoded as [cos θ, sin θ] pairs (no wrap-around discontinuity)
    * NaN frames are flagged and forward-filled

Output: (T, 24) feature matrix, dtype float32.
    12 dims — resident ego-centric joints (6 kp × xy)
     6 dims — intruder, in resident frame: nose(xy), neck(xy), tail_base(xy)
     6 dims — inter-animal:
              distance(nose0, nose1), distance(nose0, tail1),
              distance(COM0, COM1), cos(Δθ), sin(Δθ), approach_speed

Notes:
    * fps used only for approach_speed (defaults 30, CalMS21 rig)
    * input shape: (T, 2_mice, 2_xy, 7_kp) — CalMS21 raw layout
    * we keep tail (joint 6) for both mice here — needed for body axis;
      tail is dropped only from the joint columns, not from axis estimation
"""
from __future__ import annotations

import numpy as np

EPS = 1e-3
KP = {"nose": 0, "left_ear": 1, "right_ear": 2, "neck": 3,
      "left_hip": 4, "right_hip": 5, "tail_base": 6}
NUM_FEATURES = 24


def _body_axis(kp_TKD: np.ndarray) -> np.ndarray:
    """Body axis vector tail_base → neck per frame. Shape: (T, 2)."""
    return kp_TKD[:, KP["neck"], :] - kp_TKD[:, KP["tail_base"], :]


def _safe_angle(v: np.ndarray) -> np.ndarray:
    """Return (cos θ, sin θ) for vectors v of shape (T, 2).
    Degenerate (norm < EPS) rows are NaN — caller forward-fills.
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.where(norm > EPS, v / np.maximum(norm, EPS),
                   np.full_like(v, np.nan))
    return out  # (T, 2) where col0=cos, col1=sin


def _rotation_matrices(cos_sin: np.ndarray) -> np.ndarray:
    """Build rotation matrices that map body axis to +y.

    If body axis unit vector is (c, s) with current angle θ = atan2(s, c),
    we want R such that R · (c, s)^T = (0, 1)^T. That is rotation by
    (π/2 − θ), giving matrix:
        R = [[ sin θ,  -cos θ ],
             [ cos θ,   sin θ ]]

    Args:
        cos_sin: (T, 2) unit vectors. NaN rows produce identity rotations
                 (handled by forward-fill upstream).

    Returns:
        (T, 2, 2) rotation matrices.
    """
    c = cos_sin[:, 0]
    s = cos_sin[:, 1]
    R = np.stack([
        np.stack([s, -c], axis=-1),
        np.stack([c,  s], axis=-1),
    ], axis=-2)
    return R  # (T, 2, 2)


def _forward_fill_nan(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Forward-fill rows containing any NaN. Returns (filled, frac_nan).
    Only call-sites pass 2-D (T, D) arrays — single-row backward-fill at the
    start by selecting the first valid row.
    """
    assert arr.ndim == 2, f"expected 2-D, got {arr.shape}"
    nan_mask = np.isnan(arr).any(axis=1)
    frac = float(nan_mask.mean())
    if not nan_mask.any():
        return arr, 0.0
    valid_idx = np.where(~nan_mask)[0]
    if valid_idx.size == 0:
        raise ValueError("all rows are NaN — cannot forward-fill")
    idx_map = np.searchsorted(valid_idx, np.arange(len(arr)), side="right") - 1
    idx_map = np.clip(idx_map, 0, len(valid_idx) - 1)
    return arr[valid_idx[idx_map]], frac


def ego_centric_dyadic(
    kp_TMDK: np.ndarray,
    fps: float = 30.0,
    *,
    resident: int = 0,
) -> tuple[np.ndarray, dict]:
    """Convert CalMS21 raw keypoints → 24-dim ego-centric dyadic features.

    Args:
        kp_TMDK: (T, 2_mice, 2_xy, 7_kp) raw CalMS21 layout
        fps: frame rate (used for approach speed)
        resident: index of resident mouse (the ego frame is anchored to it)

    Returns:
        (features, info):
            features: (T, 24) float32
            info: dict with frac_nan_axis (degeneracy rate)
    """
    arr = np.asarray(kp_TMDK, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1:3] != (2, 2) or arr.shape[3] != 7:
        raise ValueError(f"expected (T,2,2,7), got {arr.shape}")

    # (T, 2_mice, 7_kp, 2_xy)  joint-major layout
    arr = arr.transpose(0, 1, 3, 2)
    res = arr[:, resident, :, :]                            # (T, 7, 2)
    intr = arr[:, 1 - resident, :, :]                       # (T, 7, 2)

    com_res = res.mean(axis=1, keepdims=True)               # (T, 1, 2)
    com_intr = intr.mean(axis=1, keepdims=True)

    # 1. body axis of resident
    axis = _body_axis(res)                                  # (T, 2)
    axis_unit_raw = _safe_angle(axis)                       # (T, 2) with NaN for degenerate
    axis_unit, frac_nan_axis = _forward_fill_nan(axis_unit_raw)
    R = _rotation_matrices(axis_unit)                       # (T, 2, 2)

    # 2. translate then rotate resident to ego frame
    res_centred = res - com_res                             # (T, 7, 2)
    res_ego = np.einsum("tij,tkj->tki", R, res_centred)     # (T, 7, 2)

    # 3. intruder in resident frame
    intr_centred = intr - com_res                           # subtract resident COM
    intr_ego = np.einsum("tij,tkj->tki", R, intr_centred)   # (T, 7, 2)

    # 4. inter-animal features
    nose0 = res_ego[:, KP["nose"], :]
    tail0 = res_ego[:, KP["tail_base"], :]
    com0_ego = res_ego.mean(axis=1)                         # (T, 2)
    com1_ego = intr_ego.mean(axis=1)                        # (T, 2)
    nose1 = intr_ego[:, KP["nose"], :]
    tail1 = intr_ego[:, KP["tail_base"], :]

    d_nose_nose = np.linalg.norm(nose0 - nose1, axis=-1)
    d_nose_tail = np.linalg.norm(nose0 - tail1, axis=-1)
    d_com_com = np.linalg.norm(com0_ego - com1_ego, axis=-1)

    # relative orientation: intruder's body axis in res frame, encoded as cos/sin
    intr_axis = intr_ego[:, KP["neck"], :] - intr_ego[:, KP["tail_base"], :]
    intr_axis_unit_raw = _safe_angle(intr_axis)
    intr_axis_unit, _ = _forward_fill_nan(intr_axis_unit_raw)
    # in ego frame the resident axis is exactly (0, 1). For the signed angle Δθ from
    # (0,1) to the intruder axis (cosα, sinα): cos Δθ = dot = sinα = intr_axis_y;
    # sin Δθ = cross((0,1),(cosα,sinα)) = -cosα = -intr_axis_x (handedness matters).
    cos_dtheta = intr_axis_unit[:, 1]
    sin_dtheta = -intr_axis_unit[:, 0]

    # approach speed: d(d_com_com)/dt (negative = approaching)
    d_com_pad = np.concatenate([d_com_com[:1], d_com_com])
    approach_speed = -np.diff(d_com_pad) * fps

    # 5. assemble — keep only the 6 non-tail joints for clean dim count
    keep = [KP["nose"], KP["left_ear"], KP["right_ear"], KP["neck"],
            KP["left_hip"], KP["right_hip"]]
    res_kp_flat = res_ego[:, keep, :].reshape(arr.shape[0], 12)   # 12
    intr_subset = intr_ego[:, [KP["nose"], KP["neck"], KP["tail_base"]], :]
    intr_kp_flat = intr_subset.reshape(arr.shape[0], 6)            # 6
    inter = np.stack([d_nose_nose, d_nose_tail, d_com_com,
                      cos_dtheta, sin_dtheta, approach_speed], axis=-1)   # 6

    feats = np.concatenate([res_kp_flat, intr_kp_flat, inter], axis=-1).astype(np.float32)
    assert feats.shape == (arr.shape[0], NUM_FEATURES), feats.shape

    info = {"frac_nan_axis": frac_nan_axis,
            "num_features": NUM_FEATURES,
            "feature_blocks": {
                "resident_ego_xy": (0, 12),
                "intruder_ego_xy_nose_neck_tail": (12, 18),
                "inter_animal": (18, 24)}}
    return feats, info


__all__ = ["ego_centric_dyadic", "NUM_FEATURES"]
