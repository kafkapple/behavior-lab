"""Generate full 18000-frame 3D keypoint dataset for both predictors + benchmark.

Steps:
1. SA SA_MAP corrected from actual h5 inspection (12 anatomical overlaps).
2. Triangulate ALL 18000 video frames per model (RN50 + SA zero-shot).
3. Save full_*.npz with shape (18000, 22, 3) + valid_mask.
4. Benchmark vs MAMMAL pseudo-GT (3600 frames on 5-step grid) + Li GT (81).
5. Re-save results.csv + per_kp_error.csv.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/joon/dev/behavior-lab-kp/src")
sys.path.insert(0, "/home/joon/dev/behavior-lab-kp/scripts")
from pathlib import Path
import numpy as np
import pandas as pd
from render_kp_overlay import load_cameras

DATA = Path("/home/joon/dev/behavior-lab-kp/data")
PRED = Path("/node_data/joon/behavior-lab-kp-benchmark/predictions")
OUT = Path("/home/joon/dev/behavior-lab-kp/outputs/kp_benchmark")
OUT.mkdir(parents=True, exist_ok=True)
PROB_MIN = 0.10
N_FRAMES = 18000

MAMMAL_KP = ["L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
             "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow",
             "L_shoulder", "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
             "L_foot", "L_knee", "L_hip", "R_foot", "R_knee", "R_hip"]

# DLC trained ResNet50 — bodyparts match MAMMAL 1:1
RN_MAP = {bp: bp for bp in MAMMAL_KP}

# SA-TopViewMouse 27 → MAMMAL 22 (verified from h5 inspection)
SA_MAP = {
    "left_ear": "L_ear",
    "right_ear": "R_ear",
    "nose": "nose",
    "neck": "neck",
    "mid_back": "body_middle",
    "tail_base": "tail_root",
    "tail3": "tail_middle",
    "tail_end": "tail_end",
    "left_shoulder": "L_shoulder",
    "right_shoulder": "R_shoulder",
    "left_hip": "L_hip",
    "right_hip": "R_hip",
}
# Unmappable in SA: L_paw, L_paw_end, L_elbow, R_paw, R_paw_end, R_elbow,
#                   L_foot, L_knee, R_foot, R_knee  (10 keypoints stay NaN)

cams = load_cameras(DATA / "markerless_mouse_1/labels/label3d_dannce.mat")
Ps = [c["K"] @ np.hstack([c["R"], c["t"].reshape(3, 1)]) for c in cams]


def triangulate(pts2d, probs):
    A = []
    for P, p, q in zip(Ps, pts2d, probs):
        if not np.all(np.isfinite(p)) or q < PROB_MIN:
            continue
        x, y = p
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    if len(A) < 4:
        return np.full(3, np.nan)
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-9 else np.full(3, np.nan)


def get_xyq(df, frame_i, kp_name):
    if frame_i >= len(df):
        return np.nan, np.nan, 0.0
    scorer = df.columns[0][0]
    if df.columns.nlevels == 4:
        # (scorer, animal, kp, axis)
        for animal in ("animal0", "individual1"):
            try:
                x = float(df.iloc[frame_i][(scorer, animal, kp_name, "x")])
                y = float(df.iloc[frame_i][(scorer, animal, kp_name, "y")])
                q = float(df.iloc[frame_i][(scorer, animal, kp_name, "likelihood")])
                if np.isfinite(x) and q > 0:
                    return x, y, q
            except KeyError:
                continue
        return np.nan, np.nan, 0.0
    try:
        x = float(df.iloc[frame_i][(scorer, kp_name, "x")])
        y = float(df.iloc[frame_i][(scorer, kp_name, "y")])
        q = float(df.iloc[frame_i][(scorer, kp_name, "likelihood")])
        return x, y, q
    except KeyError:
        return np.nan, np.nan, 0.0


def triangulate_all(cam_dfs, name_map, tag):
    print(f"  triangulating {N_FRAMES} frames × 22 kp...")
    out = np.full((N_FRAMES, 22, 3), np.nan, dtype=np.float32)
    src_by_mam = {v: k for k, v in name_map.items()}
    for fi in range(N_FRAMES):
        if fi % 3000 == 0:
            print(f"    {tag}: frame {fi}/{N_FRAMES}")
        for k_idx, mam in enumerate(MAMMAL_KP):
            src = src_by_mam.get(mam)
            if src is None:
                continue
            pts2d, probs = [], []
            for cam_i in range(6):
                x, y, q = get_xyq(cam_dfs[cam_i], fi, src)
                pts2d.append(np.array([x, y]))
                probs.append(q)
            out[fi, k_idx] = triangulate(pts2d, probs)
    return out


def process_model(analyze_dir, tag, name_map):
    print(f"\n=== {tag} ===")
    cam_dfs = []
    for cam_i in range(6):
        h5s = sorted(analyze_dir.glob(f"{cam_i}*.h5"))
        h5s = [h for h in h5s if "meta" not in h.name.lower()]
        if not h5s:
            print(f"  cam{cam_i}: NO h5"); return None
        cam_dfs.append(pd.read_hdf(h5s[0]))
    full = triangulate_all(cam_dfs, name_map, tag)
    valid_mask = ~np.isnan(full).any(axis=-1)  # (18000, 22)
    n_frames_full = int((valid_mask.sum(axis=-1) == 22).sum())
    n_kp_avg = float(valid_mask.mean())
    print(f"  full kp dataset: {full.shape}, fully-valid frames {n_frames_full}/{N_FRAMES}, mean kp coverage {n_kp_avg:.2f}")

    np.savez(
        OUT / f"{tag}_full_kp.npz",
        keypoints_3d=full,
        valid_mask=valid_mask,
        frame_ids=np.arange(N_FRAMES, dtype=np.int64),
        keypoint_names=np.array(MAMMAL_KP),
    )
    print(f"  saved → {OUT}/{tag}_full_kp.npz")
    return full


rn_full = process_model(PRED / "dlc_resnet50_imagenet_analyze",
                        "dlc_resnet50_imagenet", RN_MAP)
sa_full = process_model(PRED / "dlc_superanimal_zeroshot_hrnet_w32_analyze",
                        "dlc_superanimal_zeroshot_hrnet_w32", SA_MAP)


# ------------------------------ Benchmark ----------------------------------- #

def nan_root_relative_err(pred, gt, root_idx=4):
    pr = pred - pred[:, root_idx:root_idx + 1, :]
    gr = gt - gt[:, root_idx:root_idx + 1, :]
    return np.linalg.norm(pr - gr, axis=-1)


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    a = (1.0 - ci) / 2.0
    return (float(vals.mean()),
            float(np.quantile(means, a)),
            float(np.quantile(means, 1.0 - a)),
            int(vals.size))


# Load GT references
mam = np.load(DATA / "mammal_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
mam_kp = mam["keypoints"].astype(np.float64)
mam_idx = mam["frame_indices"]
li = np.load(DATA / "markerless_mouse_1/labels/li_m1_gt.npz")
li_kp = li["keypoints_3d"].astype(np.float64)
li_ids = li["frame_ids"]

print("\n=== benchmark vs MAMMAL pseudo-GT (3600 fr on 5-step grid) + Li GT (81) ===")
results, per_kp_rows = [], []
for tag, full in [("dlc_resnet50_imagenet", rn_full),
                  ("dlc_superanimal_zeroshot_hrnet_w32", sa_full)]:
    # MAMMAL eval: at mam_idx frames (3600 frames where GT exists)
    pred_at_mam = full[mam_idx]                  # (3600, 22, 3)
    err = nan_root_relative_err(pred_at_mam.astype(np.float64), mam_kp)
    pf = np.nanmean(err, axis=-1)
    mean, lo, hi, n = bootstrap_ci(pf)
    results.append({"predictor": tag, "split": "mammal_full_3600", "n_total": 3600,
                    "n_valid_frames": int(np.sum(np.isfinite(pf))),
                    "mpjpe_mean_mm": mean, "mpjpe_ci_lo": lo, "mpjpe_ci_hi": hi,
                    "kp_coverage": float(np.mean(np.isfinite(err)))})
    print(f"{tag:>40s} | mammal_full_3600 | N={n:>4d} | MPJPE={mean:7.2f} [{lo:6.2f},{hi:6.2f}] | cov={results[-1]['kp_coverage']:.2f}")
    for k, name in enumerate(MAMMAL_KP):
        v = err[:, k][np.isfinite(err[:, k])]
        per_kp_rows.append({"predictor": tag, "split": "mammal_full_3600", "kp_idx": k,
                            "kp_name": name,
                            "mpjpe_mean_mm": float(v.mean()) if v.size else float("nan"),
                            "n_valid": int(v.size)})

    # Li eval: at li_ids
    pred_at_li = full[li_ids]
    err = nan_root_relative_err(pred_at_li.astype(np.float64), li_kp)
    pf = np.nanmean(err, axis=-1)
    mean, lo, hi, n = bootstrap_ci(pf)
    results.append({"predictor": tag, "split": "li_external", "n_total": 81,
                    "n_valid_frames": int(np.sum(np.isfinite(pf))),
                    "mpjpe_mean_mm": mean, "mpjpe_ci_lo": lo, "mpjpe_ci_hi": hi,
                    "kp_coverage": float(np.mean(np.isfinite(err)))})
    print(f"{tag:>40s} | li_external      | N={n:>4d} | MPJPE={mean:7.2f} [{lo:6.2f},{hi:6.2f}] | cov={results[-1]['kp_coverage']:.2f}")
    for k, name in enumerate(MAMMAL_KP):
        v = err[:, k][np.isfinite(err[:, k])]
        per_kp_rows.append({"predictor": tag, "split": "li_external", "kp_idx": k,
                            "kp_name": name,
                            "mpjpe_mean_mm": float(v.mean()) if v.size else float("nan"),
                            "n_valid": int(v.size)})

df = pd.DataFrame(results); df.to_csv(OUT / "results.csv", index=False)
pd.DataFrame(per_kp_rows).to_csv(OUT / "per_kp_error.csv", index=False)
print(f"\n=== saved ===\n  {OUT}/results.csv\n  {OUT}/per_kp_error.csv")
print(df.to_string(index=False))
