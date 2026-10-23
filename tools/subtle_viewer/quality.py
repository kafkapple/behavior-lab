"""Label-free quality metrics of the reconstructed 3D skeleton, per session. There is no ground truth yet, so these are consistency checks, not accuracy:
a bad value flags a session; a good value does not prove the pose is correct. See docs/dev/subtle_3d_quality_plan.md. No web code."""
from __future__ import annotations

from pathlib import Path

import csv
import subprocess
import time

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import core

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONF = 0.25          # recon3D keeps a 2D point only above this confidence (subtle_pipeline_io.md)
JUMP_BODY = 0.5      # a node moving more than this fraction of the body length in one frame (50 ms) is treated as a jump


def copied_mask(xyz: np.ndarray) -> np.ndarray:
    """(T,) True where at least one node equals its position in the previous frame in all three coordinates (recon3D copies the last value when it
    cannot triangulate). A real animal does not hold a float coordinate exactly still, so an exact repeat is a copy. Frame 0 is False."""
    m = np.zeros(len(xyz), bool)
    m[1:] = (np.diff(xyz, axis=0) == 0).all(2).any(1)
    return m


def session_quality(s: dict) -> dict:
    x = s["xyz"].astype(np.float64)
    T, K = len(x), x.shape[1]
    same = np.zeros((T, K), bool); same[1:] = (np.diff(x, axis=0) == 0).all(2)
    out = {"frames": T, "copied_frames": float(same.any(1).mean()), "copied_nodes": float(same.mean())}
    L = np.array([np.linalg.norm(x[:, b] - x[:, a], axis=1) for a, b in s["edges"]])           # (edges, T) bone lengths
    cv = L.std(1) / L.mean(1)
    head = [i for i, (a, b) in enumerate(s["edges"]) if {s["nodes"][a], s["nodes"][b]} & {"nose1", "earL1", "earR1"}]   # head edges are nearly rigid; limbs and spine are not
    out["head_bone_cv"] = float(np.median(cv[head])) if head else None                          # large = the 3D head points are noisy
    out["bone_cv_all"] = float(np.median(cv))
    body = float(np.median(np.linalg.norm(x[:, s["nodes"].index("neck1")] - x[:, s["nodes"].index("tailstart1")], axis=1))) if {"neck1", "tailstart1"} <= set(s["nodes"]) else float(np.median(L))
    step = np.linalg.norm(np.diff(x, axis=0), axis=2)                                           # (T-1, nodes) displacement per frame
    out["jump_frames"] = float((step.max(1) > JUMP_BODY * body).mean())
    if s["kp"] is not None:
        ok = (s["kp"][..., 2] > CONF).sum(1)                                                    # (T, nodes): cameras above the confidence line
        out["tri_support"] = float((ok >= 2).mean())                                           # node-frames with at least 2 cameras: triangulation possible
        out["cams_mean"] = float(ok.mean())
        out["unsupported_vs_copied"] = float(((ok < 2)[1:] & same[1:]).sum() / max(1, (ok < 2)[1:].sum()))   # of node-frames with <2 cameras, share that are copies
        res = []
        for cam in range(s["kp"].shape[1]):
            try:
                res.append(core.cam_view(s, cam)["resid_px"])
            except ValueError:
                res.append(None)
        out["reproj_px"] = res                                                                  # weak-perspective residual per camera (not a calibrated reprojection error)
    return out


def project_quality(root: Path, proj: dict) -> dict:
    rows = []
    for g, sess in proj.get("groups", {}).items():
        for r in sess:
            rows.append({"group": g, "session": Path(r).stem, "path": r, **session_quality(core.load(root, r))})
    return {"rows": rows, "note": "consistency checks without ground truth; see subtle_3d_quality_plan.md"}


FF = "/opt/homebrew/bin/ffmpeg"
GAP, HALF = 40, 10        # worst frames are at least GAP frames apart; each snippet shows HALF frames either side


def worst_frames(s: dict, n: int) -> list[dict]:
    """The n frames where one node moves the most in a single step (in body lengths), with that node. Heuristic ranking, not a verdict."""
    x = s["xyz"].astype(np.float64)
    body = float(np.median(np.linalg.norm(x[:, s["nodes"].index("neck1")] - x[:, s["nodes"].index("tailstart1")], axis=1)))
    step = np.linalg.norm(np.diff(x, axis=0), axis=2) / body
    score, node = step.max(1), step.argmax(1)
    out = []
    for f in np.argsort(-score):
        if all(abs(int(f) + 1 - o["frame"]) > GAP for o in out) and HALF < f + 1 < len(x) - HALF:
            out.append({"frame": int(f) + 1, "jump_body": float(score[f]), "node": int(node[f])})
        if len(out) == n:
            break
    return out


def worst_clips(root: Path, proj: dict, name: str, per_session: int = 3) -> Path:
    """One clip per session that has a proxy video: the worst frames, 2D keypoints (green) next to the 3D skeleton projected into the same
    camera by the weak-perspective fit (magenta). The flagged node is ringed. A person rates each snippet; the csv has the rating column."""
    out = root / ".subtle_results" / name / f"{time.strftime('%y%m%d_%H%M%S')}_quality"
    out.mkdir(parents=True)
    cams = {s["path"]: s["cams"] for s in core.scan(root)}
    font = ImageFont.load_default(size=15)
    rows = []
    for g, sess in proj["groups"].items():
        for r in sess:
            if not cams.get(r):
                continue
            s, cam = core.load(root, r), int(sorted(cams[r])[0])
            vp = root / cams[r][str(cam)] if str(cam) in cams[r] else root / cams[r][sorted(cams[r])[0]]
            P = np.array(core.cam_view(s, cam)["P"]); cw, ch = core.CELLS[cam][2:]
            wd, ht = map(int, subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(vp)], capture_output=True, text=True, check=True).stdout.strip().split("x"))
            kp = core.kp_cell(s, cam)
            proj3 = (np.c_[s["xyz"].reshape(-1, 3), np.ones(len(s["xyz"]) * s["xyz"].shape[1])] @ P).reshape(len(s["xyz"]), -1, 2) / [cw, ch] * [wd, ht]
            enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{wd}x{ht}", "-r", "20", "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(out / f"worst_{Path(r).stem.split('_')[0]}.mp4")], stdin=subprocess.PIPE)
            for w in worst_frames(s, per_session):
                st = w["frame"] - HALF
                raw = subprocess.run([FF, "-v", "error", "-ss", f"{st / 20:.3f}", "-i", str(vp), "-frames:v", str(2 * HALF + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
                fr = np.frombuffer(raw, np.uint8).reshape(-1, ht, wd, 3)
                d2 = float(np.linalg.norm(kp[w["frame"], w["node"], :2] * [wd, ht] - proj3[w["frame"], w["node"]]))
                rows.append({"group": g, "session": Path(r).stem.split("_")[0], "frame": w["frame"], "jump_body": round(w["jump_body"], 2), "node": s["nodes"][w["node"]], "disagree_px": round(d2, 1), "rating": ""})
                for i, f in enumerate(fr):
                    im = Image.fromarray(f.copy()); d = ImageDraw.Draw(im); t = st + i
                    for a, b in s["edges"]:
                        d.line([*proj3[t, a], *proj3[t, b]], fill=(255, 0, 255), width=2)
                        if kp[t, a, 2] > .25 and kp[t, b, 2] > .25:
                            d.line([kp[t, a, 0] * wd, kp[t, a, 1] * ht, kp[t, b, 0] * wd, kp[t, b, 1] * ht], fill=(46, 204, 113), width=2)
                    x0, y0 = proj3[t, w["node"]]
                    if i == HALF:
                        d.ellipse([x0 - 9, y0 - 9, x0 + 9, y0 + 9], outline=(255, 0, 0), width=3)
                    d.rectangle([0, 0, wd, 22], fill=(0, 0, 0)); d.text((4, 3), f"{g} {rows[-1]['session']} frame {t} | jump {w['jump_body']:.2f} body @ {rows[-1]['node']} | green=2D magenta=3D", fill=(255, 255, 255), font=font)
                    enc.stdin.write(np.asarray(im).tobytes())
            enc.stdin.close(); enc.wait()
    with open(out / "worst_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "session", "frame", "jump_body", "node", "disagree_px", "rating"]); w.writeheader(); w.writerows(rows)
    return out


def figure(R: dict, out: Path) -> None:
    """Per-session consistency metrics, coloured by group, with the group means; saved as quality.png."""
    rows = R["rows"]
    cols = [("copied_frames", "frames with a copied node (%)", 100), ("jump_frames", "frames with a jump over half a body length (%)", 100), ("head_bone_cv", "head bone length CV", 1), ("tri_support", "node-frames with 2+ cameras (%)", 100)]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    col = {g: c for g, c in zip(dict.fromkeys(r["group"] for r in rows), ("#5ac8fa", "#ff9f0a", "#999"))}
    for a, (k, t, m) in zip(ax, cols):
        a.bar(range(len(rows)), [m * r[k] for r in rows], color=[col[r["group"]] for r in rows])
        a.set(xticks=range(len(rows)), xticklabels=[r["session"].split("_")[0] for r in rows], title=t); a.tick_params(axis="x", rotation=60)
        for g, c in col.items():
            v = [m * r[k] for r in rows if r["group"] == g]; idx = [i for i, r in enumerate(rows) if r["group"] == g]
            a.hlines(np.mean(v), min(idx) - .4, max(idx) + .4, color="k", ls="--", lw=1)
    fig.suptitle("3D skeleton consistency checks per session (no ground truth); dashed = group mean", fontsize=11); fig.tight_layout(); fig.savefig(out / "quality.png", dpi=110); plt.close(fig)
