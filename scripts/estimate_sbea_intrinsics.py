"""
SBeA camera intrinsics estimation: 3 methods compared.

Method 1: Direct decomposition of cam_mat_all from caliParas.mat
Method 2: Arena geometry via HoughCircles (known physical arena diameter)
Method 3: Epipolar self-calibration via SIFT F-matrix + focal length sweep
"""

import subprocess, sys, json, os, cv2
import numpy as np
import scipy.io

# 260814: gpu03 절대경로(/node_data_2/joon/…)였던 것을 env 경유로 바꿨다. 서버 종료로 구 경로는 죽었다.
#   SBEA_DIR=/mnt/d/data/raw/SBeA/individual SBEA_SESSION=rec1-M7-20221108 python scripts/estimate_sbea_intrinsics.py
_DIR = os.environ.get("SBEA_DIR", "/mnt/d/data/raw/SBeA/individual")
_SESSION = os.environ.get("SBEA_SESSION", "rec1-M7-20221108")
MAT_PATH = f"{_DIR}/{_SESSION}-caliParas.mat"
VID_BASE = f"{_DIR}/{_SESSION}-camera-{{}}.mp4"
ARENA_DIAMETER_MM = 300.0   # cam1 frame shows "30cm" ruler → arena ~300mm
IMG_W, IMG_H = 1288, 964
N_CAMS = 4


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_frame(vid_path, t_sec=5.0):
    cmd = ["ffmpeg", "-ss", str(t_sec), "-i", vid_path,
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
           "-loglevel", "error"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    buf = np.frombuffer(r.stdout, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_mat():
    d = scipy.io.loadmat(MAT_PATH)
    cp = d["caliParams"][0, 0]
    cma = np.array(cp["cam_mat_all"])       # (3,4,4) Python = MATLAB[4,4,3]
    R_ref = cp["rotation"][0, 0]
    t_ref = cp["translation"][0, 0].ravel()
    C_ref = -R_ref.T @ t_ref
    return cma, R_ref, t_ref, C_ref


# ── Method 1: cam_mat_all decomposition ──────────────────────────────────────

def method1_decompose():
    cma, R_ref, t_ref, C_ref = load_mat()
    results = []

    for cam_id in range(N_CAMS):
        P = cma[:, cam_id, :].astype(np.float64)
        # Normalize so that ||m3[:3]|| = 1
        scale = np.linalg.norm(P[2, :3])
        if scale < 1e-9:
            results.append({"cam_id": cam_id, "status": "scale~0 (reference camera at origin)"})
            continue
        P_n = P / scale
        try:
            K, R, t_h, *_ = cv2.decomposeProjectionMatrix(P_n)
            K /= K[2, 2]
            t = (t_h[:3] / t_h[3]).ravel()
            C = -R.T @ t
            fx, fy = float(K[0, 0]), float(K[1, 1])
            physical = (200 < fx < 6000) and (200 < fy < 6000) and (abs(fx - fy) / max(fx, fy) < 0.3)
            results.append({
                "cam_id": cam_id, "fx": round(fx, 1), "fy": round(fy, 1),
                "cx": round(float(K[0, 2]), 1), "cy": round(float(K[1, 2]), 1),
                "dist_mm": round(float(np.linalg.norm(C)), 1),
                "physical": physical,
            })
        except Exception as e:
            results.append({"cam_id": cam_id, "status": f"decompose failed: {e}"})

    # Reference camera info
    dist_ref = float(np.linalg.norm(C_ref))
    return {"method": "cam_mat_all decomposition",
            "ref_cam_dist_mm": round(dist_ref, 1),
            "cameras": results}


# ── Method 2: Arena geometry ─────────────────────────────────────────────────

def detect_arena_circle(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=200, param1=60, param2=35,
        minRadius=150, maxRadius=600,
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    # Return the largest circle (most likely the arena boundary)
    best = sorted(circles, key=lambda c: -c[2])[0]
    return best   # (cx_px, cy_px, r_px)


def method2_arena_geometry():
    """
    Arena is circular with known diameter = ARENA_DIAMETER_MM.
    Cameras look inward at arena center from ~640mm away (horizontal).
    For camera at position C with distance d_plane = horizontal dist to arena center:
        fx ≈ (r_px * d_plane) / r_mm
    where r_px = arena radius in pixels, r_mm = ARENA_DIAMETER_MM/2.
    d_plane estimated from camera center ||C|| ≈ 640mm.
    """
    _, _, _, C_ref = load_mat()
    d_plane = float(np.linalg.norm(C_ref))  # ~640mm (horizontal to arena center)

    cam_results = []
    for cam_id in range(N_CAMS):
        vid = VID_BASE.format(cam_id)
        img = extract_frame(vid, t_sec=5.0)
        if img is None:
            cam_results.append({"cam_id": cam_id, "status": "video read failed"})
            continue

        circle = detect_arena_circle(img)
        if circle is None:
            cam_results.append({"cam_id": cam_id, "status": "arena circle not detected"})
            continue

        cx_px, cy_px, r_px = circle
        arena_r_mm = ARENA_DIAMETER_MM / 2.0
        fx_est = (r_px * d_plane) / arena_r_mm    # fx = r_px * D / R_mm
        fy_est = fx_est   # assume square pixels
        cam_results.append({
            "cam_id": cam_id,
            "arena_r_px": int(r_px), "arena_cx_px": int(cx_px), "arena_cy_px": int(cy_px),
            "fx": round(fx_est, 1), "fy": round(fy_est, 1),
            "cx_est": int(cx_px), "cy_est": int(cy_px),
            "d_plane_mm": round(d_plane, 1),
        })

    valid = [c for c in cam_results if "fx" in c]
    summary = {}
    if valid:
        fxs = [c["fx"] for c in valid]
        summary = {"mean_fx": round(np.mean(fxs), 1), "std_fx": round(np.std(fxs), 1),
                   "n_detected": len(valid)}

    return {"method": "arena geometry (HoughCircles)",
            "arena_diameter_mm": ARENA_DIAMETER_MM,
            "cameras": cam_results, **summary}


# ── Method 3: Epipolar self-calibration ──────────────────────────────────────

def _orb_matches(img_a, img_b):
    """ORB-based matching with CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ga = clahe.apply(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY))
    gb = clahe.apply(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY))
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=10)
    kpa, desa = orb.detectAndCompute(ga, None)
    kpb, desb = orb.detectAndCompute(gb, None)
    if desa is None or desb is None or len(kpa) < 15 or len(kpb) < 15:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desa, desb, k=2)
    good = [m for m, n in raw if m.distance < 0.85 * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def _sift_matches(img_a, img_b):
    """SIFT matching with CLAHE; more discriminative than ORB."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ga = clahe.apply(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY))
    gb = clahe.apply(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY))
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    kpa, desa = sift.detectAndCompute(ga, None)
    kpb, desb = sift.detectAndCompute(gb, None)
    if desa is None or desb is None or len(kpa) < 15 or len(kpb) < 15:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(desa, desb, k=2)
    good = [m for m, n in raw if m.distance < 0.80 * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def _sweep_focal(pts0, pts1, F):
    """Return (best_f, best_score) via singular-value sweep on E = K^T F K.

    Valid E: s0 ≈ s1 >> s2 ≈ 0.
    Score = balance(s0,s1) / (s2/s0 + 1e-6).  Higher = better.
    """
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    best_f, best_score = None, -1.0
    for f_try in np.arange(600, 3000, 30):
        K = np.array([[f_try, 0, cx], [0, f_try, cy], [0, 0, 1.0]])
        E = K.T @ F @ K
        s = np.linalg.svd(E, compute_uv=False)
        if s[0] < 1e-9:
            continue
        balance = min(s[0], s[1]) / max(s[0], s[1]) if s[1] > 1e-9 else 0
        rank2_quality = balance / (s[2] / s[0] + 1e-6)   # large when s2≈0
        if rank2_quality > best_score:
            best_score = rank2_quality
            best_f = float(f_try)
    return best_f, best_score


def _epipolar_pair(img_a, img_b, t_tag=""):
    """Run ORB+SIFT match + F + focal sweep for one camera pair."""
    pts0, pts1 = _orb_matches(img_a, img_b)
    if len(pts0) < 15:
        pts0, pts1 = _sift_matches(img_a, img_b)
    if len(pts0) < 15:
        return None, f"too few ORB matches: {len(pts0)}"
    F, mask_f = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 1.5, 0.99)
    if F is None or mask_f is None:
        return None, "F matrix failed"
    inliers = int(mask_f.sum())
    if inliers < 10:
        return None, f"F inliers too low: {inliers}"
    pts0_in = pts0[mask_f.ravel() == 1]
    pts1_in = pts1[mask_f.ravel() == 1]
    best_f, score = _sweep_focal(pts0_in, pts1_in, F)
    if best_f is None:
        return None, "focal sweep degenerate"
    return {"n_matches": len(pts0), "F_inliers": inliers,
            "best_f": round(best_f, 1), "sv_score": round(score, 1)}, None


def method3_epipolar():
    """
    ORB matching across multiple camera pairs (0-1, 0-3, 1-2, 2-3) and
    multiple time points; aggregate focal-length votes.
    """
    T_SECS = [3.0, 5.0, 10.0, 20.0]
    PAIRS = [(0, 1), (1, 2), (2, 3), (0, 3)]   # adjacent + diagonal

    # Load frames
    all_imgs = {}
    for cam_id in range(N_CAMS):
        frames = []
        for t in T_SECS:
            img = extract_frame(VID_BASE.format(cam_id), t_sec=t)
            if img is not None:
                frames.append(img)
        all_imgs[cam_id] = frames

    votes = []   # list of best_f from valid pairs × time points
    pair_results = []

    for (ca, cb) in PAIRS:
        frames_a = all_imgs.get(ca, [])
        frames_b = all_imgs.get(cb, [])
        if not frames_a or not frames_b:
            pair_results.append({"pair": f"{ca}-{cb}", "status": "video missing"})
            continue
        pair_votes = []
        for img_a, img_b in zip(frames_a, frames_b):
            res, err = _epipolar_pair(img_a, img_b)
            if res is not None:
                pair_votes.append(res["best_f"])
        if pair_votes:
            med_f = float(np.median(pair_votes))
            pair_results.append({
                "pair": f"{ca}-{cb}",
                "n_time_votes": len(pair_votes),
                "median_f": round(med_f, 1),
                "votes": [round(v, 1) for v in pair_votes],
            })
            votes.extend(pair_votes)
        else:
            pair_results.append({"pair": f"{ca}-{cb}", "status": "all time points failed"})

    if not votes:
        return {"method": "epipolar self-calibration",
                "status": "no valid pairs", "pairs": pair_results}

    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    median_f = float(np.median(votes))
    std_f = float(np.std(votes))
    # Degenerate if: std > 30% of median, or median at sweep boundary (600px)
    degenerate = (std_f > 0.30 * median_f) or (median_f < 650)
    result = {
        "method": "epipolar self-calibration",
        "n_votes": len(votes),
        "median_f": round(median_f, 1),
        "mean_f": round(float(np.mean(votes)), 1),
        "std_f": round(std_f, 1),
        "pairs": pair_results,
    }
    if degenerate:
        result["status"] = (
            f"degenerate (cylindrical scene — F matrix ill-conditioned; "
            f"median={round(median_f,1)}px at sweep boundary, std={round(std_f,1)}px)"
        )
    else:
        result.update({"fx": round(median_f, 1), "fy": round(median_f, 1),
                       "cx": round(cx, 1), "cy": round(cy, 1)})
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== SBeA Intrinsics Estimation ===\n")

    r1 = method1_decompose()
    r2 = method2_arena_geometry()
    r3 = method3_epipolar()

    results = {"method1": r1, "method2": r2, "method3": r3}

    print(json.dumps(results, indent=2))

    # Summary comparison table
    print("\n=== Comparison Table ===")
    print(f"{'Method':<35} {'fx (px)':<12} {'fy (px)':<12} {'Note'}")
    print("-" * 75)

    # Method 1: best physical result (if any)
    m1_physical = [c for c in r1["cameras"] if c.get("physical")]
    if m1_physical:
        c = m1_physical[0]
        print(f"{'M1: cam_mat_all decompose (cam'+str(c['cam_id'])+')':<35} {c['fx']:<12} {c['fy']:<12} dist={c['dist_mm']}mm (non-physical!)")
    else:
        print(f"{'M1: cam_mat_all decompose':<35} {'N/A':<12} {'N/A':<12} all decompositions non-physical")

    # Method 2: mean across detected cameras
    if "mean_fx" in r2:
        print(f"{'M2: arena geometry (HoughCircles)':<35} {r2['mean_fx']:<12} {r2['mean_fx']:<12} n={r2['n_detected']} cameras, arena={ARENA_DIAMETER_MM}mm dia")
    else:
        print(f"{'M2: arena geometry':<35} {'N/A':<12} {'N/A':<12} arena circle not detected")

    # Method 3: epipolar
    if "fx" in r3:
        print(f"{'M3: epipolar self-calib (ORB)':<35} {r3['fx']:<12} {r3['fy']:<12} "
              f"n_votes={r3['n_votes']}, std={r3['std_f']}px")
    else:
        print(f"{'M3: epipolar self-calib':<35} {'N/A':<12} {'N/A':<12} {r3.get('status','')}")

    print(f"\nExpected range for SBeA (1288x964 @~640mm): fx ≈ 1900-2200px")
    print(f"Reference cam dist from origin: {r1['ref_cam_dist_mm']}mm")


if __name__ == "__main__":
    main()
