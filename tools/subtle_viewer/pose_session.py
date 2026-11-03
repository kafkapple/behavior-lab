"""Worker (runs in tools/pose_runtime venv): predict one session's 5 camera videos with a SLEAP single-instance model, triangulate, and write a viewer session
`<stem>_sleap_<tag>.json.gz` next to the source. stdin = JSON request, stderr = progress, last stdout line = JSON result.
3D = top2 triangulation (conf>0.25) with the calibration; when the source session has skeleton3d, our 3D is mapped into its frame so both overlay and compare."""
import gzip, json, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from behavior_lab.rig.avatar import ORIGIN as _ORIGIN, SIZE as _SIZE, kp2d_cells  # noqa: E402
from behavior_lab.rig.multiview import load_calib, triangulate_sequence, _normalized  # noqa: E402
from rig_triangulation_check import similarity  # noqa: E402
import sleap_io as sio  # noqa: E402

ORIGIN = list(_ORIGIN.values())                       # rig dicts are keyed by camera name; the order is camera 1..5
COMP = (max(o[0] + z[0] for o, z in zip(ORIGIN, _SIZE.values())), max(o[1] + z[1] for o, z in zip(ORIGIN, _SIZE.values())))
assert len(ORIGIN) == 5 and COMP == (3600, 2000), (ORIGIN, COMP)


def tri_weighted(K2, cal, thr=0.25):
    """DLT over every camera with conf > thr, each camera's two rows weighted by its confidence (261103: fewer 3D jumps than the server's top-2 rule,
    same 2D: 71 -> 17 jump frames on web LP, 46 -> 17 on SLEAP 1423; held-out camera error 25.4 -> 23.5 px). ponytail: uses multiview._normalized (private)."""
    T, _, N, _ = K2.shape; X = np.full((T, N, 3), np.nan); used = np.zeros((T, N), int)
    for t in range(T):
        for k in range(N):
            c = K2[t, :, k, 2]; idx = [i for i in range(len(cal)) if c[i] > thr]
            if len(idx) < 2: continue
            rows = []
            for i in idx:
                u, P = _normalized(K2[t, i, k, :2], cal[i]), cal[i]["P"]; rows += [c[i] * (u[0] * P[2] - P[0]), c[i] * (u[1] * P[2] - P[1])]
            v = np.linalg.svd(np.array(rows))[2][-1]; X[t, k] = v[:3] / v[3]; used[t, k] = len(idx)
    return X, used


def log(m): print(m, file=sys.stderr, flush=True)


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def main(req):
    root, src, out = Path(req["root"]), Path(req["root"]) / req["session"], Path(req["out"])
    out.mkdir(parents=True, exist_ok=True)
    calib = Path(req["calib"])
    if not calib.is_file():
        raise SystemExit(f"calibration not found: {calib} (set SUBTLE_CALIB to the aniposelib toml)")
    cams = req["cams"]
    if len(cams) != 5:
        raise SystemExit(f"need 5 camera videos (stem_cam_1..5.mp4), found {len(cams)}")
    track = Path(sys.executable).parent / "sleap-nn"
    for c in range(5):
        if req.get("reuse_preds") and (out / f"pred_cam_{c + 1}.slp").is_file():
            continue                                         # test hook: skip inference when the predictions are already there
        log(f"predict cam_{c + 1}/5")
        r = subprocess.run([str(track), "track", "-i", cams[str(c)], "-m", req["model"], "-o", str(out / f"pred_cam_{c + 1}.slp"), "-d", device(), "-b", "4"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"sleap-nn failed on cam_{c + 1}: {r.stderr[-600:]}")
    labs = [sio.load_slp(str(out / f"pred_cam_{c + 1}.slp")) for c in range(5)]
    names = list(labs[0].skeleton.node_names)
    edges = [[names.index(e.source.name), names.index(e.destination.name)] for e in labs[0].skeleton.edges]
    with gzip.open(src, "rt") as f:
        base = json.load(f)
    T = len(base.get("keypoint") or base.get("skeleton3d") or {}) or 1 + max(lf.frame_idx for lab in labs for lf in lab.labeled_frames)
    kp = {str(t): {str(c): {str(k): [0.0, 0.0, 0.0] for k in range(len(names))} for c in range(5)} for t in range(T)}
    for c, lab in enumerate(labs):
        idx = [list(lab.skeleton.node_names).index(n) for n in names]
        for lf in lab.labeled_frames:
            if lf.frame_idx < T and lf.instances:
                for k, (x, y, s) in enumerate(lf.instances[0].numpy(scores=True)[idx]):
                    if np.isfinite(x) and np.isfinite(y):
                        kp[str(lf.frame_idx)][str(c)][str(k)] = [float((x + ORIGIN[c][0]) / COMP[0]), float((y + ORIGIN[c][1]) / COMP[1]), float(s)]
    res = dict(base); res["keypoint"] = kp; res["node"] = {**base.get("node", {}), "id": names, "line": edges}
    log("triangulate")
    cal, frames = load_calib(calib), list(range(T))
    method = req.get("method", "dlt_w")                      # dlt_w (default) or top2 (the server rule)
    xo, _ = tri_weighted(kp2d_cells(res, frames), cal) if method == "dlt_w" else triangulate_sequence(kp2d_cells(res, frames), cal, "top2", 0.25)
    fin = np.isfinite(xo[..., 0]); note = {"triangulation": f"{method}, conf>0.25" + (", rows weighted by confidence" if method == "dlt_w" else ""), "calib": str(calib)}
    sk = base.get("skeleton3d")
    if sk and base.get("keypoint"):
        sw = np.array([sk[str(f)] for f in frames], float)
        xw, _ = triangulate_sequence(kp2d_cells(base, frames), cal, "top2", 0.25)
        ok = np.isfinite(xw[..., 0]) & np.isfinite(sw[..., 0])
        s, Q, t = similarity(sw[ok], xw[ok]); xo = (xo - t) @ Q / s; note["mapped_into"] = req["session"]
    xo = np.where(fin[..., None], xo, np.nan); filled = 0
    for k in range(xo.shape[1]):
        good = np.where(np.isfinite(xo[:, k, 0]))[0]
        if not len(good):
            raise SystemExit(f"node {names[k]} was never triangulated")
        last = good[0]
        for i in range(T):
            if np.isfinite(xo[i, k, 0]): last = i
            else: xo[i, k] = xo[last, k]; filled += 1
    sk3 = {str(f): [[round(float(v), 4) for v in p] for p in xo[i]] for i, f in enumerate(frames)}
    note.update({"points_missing_before_fill": int((~fin).sum()), "points_filled_from_neighbors": filled})
    res.update({"model": "sleap", "weight": req["model"], "skeleton3d": sk3, "skeleton3d_postprocessing": sk3, "ours_note": note})
    dst = src.with_name(src.name.removesuffix(".json.gz").split("_sleap_")[0] + f"_sleap_{req['tag']}.json.gz")
    with gzip.open(dst, "wt") as f:
        json.dump(res, f)
    print(json.dumps({"session": str(dst.relative_to(root)), "nodes": names, "frames": T, "filled": filled}))


if __name__ == "__main__":
    main(json.load(sys.stdin))
