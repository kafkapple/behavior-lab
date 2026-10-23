"""Save the group-difference result of a level sweep: tables (csv, json), a figure (png), and video clips of the flagged clusters.
Flagged = selection-corrected significant; if none is, the top clusters by |t| are clipped and labeled exploratory. No web code."""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np

import core
import model
import sweep

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FF = "/opt/homebrew/bin/ffmpeg"
MIN_BOUT, CLIP, PER_GROUP, TOP_EXPLORATORY = 10, 60, 3, 2
COL = {0: (90, 200, 250), 1: (255, 159, 10)}            # group A blue, group B orange (same as the page)


def tables(R: dict, out: Path) -> None:
    with open(out / "sweep_levels.csv", "w", newline="") as f:
        cr = list(R["criteria"])
        w = csv.writer(f); w.writerow(["level", "n_clusters", "max_abs_t", "level_thr95"] + [f"{c}_{x}" for c in cr for x in ("z", "p_level")] + ["recommended_by_" + R["recommended"]["criterion"]])
        for o in R["levels"]:
            w.writerow([o["level"], o["k"], f"{o['max_abs_t']:.4f}", f"{o['thr95']:.4f}"] + [f"{o['criteria'][c][x]:.4f}" for c in cr for x in ("z", "p_level")] + [int(o["level"] == R["recommended"]["level"])])
        w.writerow([]); w.writerow(["criterion", "recommended_n_clusters", "z", "p_fw_selection_corrected"])
        for c, v in R["by_criterion"].items():
            w.writerow([c, v["k"], f"{v['z']:.4f}", f"{v['p_fw']:.4f}"])
    a, b = R["groups"]
    with open(out / "clusters.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["level", "n_clusters", "cluster", f"mean_{a}", f"mean_{b}", "diff", "hedges_g", "t", "p_perm_raw", "p_adj_maxT_over_levels", "flag"])
        for o in R["levels"]:
            for i, c in enumerate(o["ids"]):
                flag = "significant" if o["p_adj"][i] < 0.05 else ("raw p<.05 only" if o["p_raw"][i] < 0.05 else "")
                w.writerow([o["level"], o["k"], c, f"{o['means'][a][i]:.3f}", f"{o['means'][b][i]:.3f}", f"{o['diff'][i]:.3f}", "" if o["g"][i] is None else f"{o['g'][i]:.3f}",
                            f"{o['t'][i]:.3f}", f"{o['p_raw'][i]:.4f}", f"{o['p_adj'][i]:.4f}", flag])
    (out / "summary.json").write_text(json.dumps({k: v for k, v in R.items() if k != "levels"} | {"levels": [{k: v for k, v in o.items() if k not in ("values",)} for o in R["levels"]]}, indent=1))


def figure(R: dict, root: Path, out: Path) -> None:
    rec, lv = R["recommended"], next(o for o in R["levels"] if o["level"] == R["recommended"]["level"])
    flag = set(rec["significant"]) or set(np.array(lv["ids"])[np.argsort(-np.abs(lv["t"]))[:TOP_EXPLORATORY]].tolist())
    kind = "significant after max-T over all levels" if rec["significant"] else "exploratory: top |t|, NOT significant"
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    x = [o["k"] for o in R["levels"]]
    ax[0].bar(x, [o["max_abs_t"] for o in R["levels"]], color=["#d62728" if o["level"] == rec["level"] else "#999" for o in R["levels"]])
    ax[0].plot(x, [o["thr95"] for o in R["levels"]], "k_", ms=22, label="level null 95%")
    ax[0].axhline(R["global_thr95"], color="k", ls="--", label="all-level null 95%"); ax[0].set(xlabel="number of superclusters", ylabel="max |t| over clusters", title=f"Level sweep [{rec['criterion']}] recommended n={rec['k']}, p_fw={rec['p_fw']:.2f}"); ax[0].legend()
    a, b = R["groups"]
    for i, c in enumerate(lv["ids"]):
        for g, col, dx in ((a, "#5ac8fa", -.12), (b, "#ff9f0a", .12)):
            ax[1].scatter([i + dx] * len(lv["values"][g][i]), lv["values"][g][i], color=col, s=28)
        ax[1].text(i, ax[1].get_ylim()[1] * 0.0 + max(max(lv["values"][a][i]), max(lv["values"][b][i])) + 1, ("★" if c in rec["significant"] else ("☆" if c in flag else "")) + f"p{lv['p_adj'][i]:.2f}", ha="center", fontsize=8)
    ax[1].set(xticks=range(len(lv["ids"])), xticklabels=lv["ids"], xlabel=f"cluster (n={rec['k']})", ylabel="share of frames per animal (%)", title=f"{a} blue vs {b} orange; one dot = one animal")
    emb, lab = [], []
    for r, nl in zip(R["paths"], R["frames"]):
        s = core.load(root, r); emb.append(np.asarray(s["emb"])[:nl]); lab.append(core.pick_level(s["super"], rec["level"])[:nl])
    emb, lab = np.vstack(emb), np.concatenate(lab)
    m = np.isin(lab, list(flag))
    ax[2].scatter(*emb[~m].T, s=1, c="#cccccc")
    for c in sorted(flag):
        ax[2].scatter(*emb[lab == c].T, s=2, color=plt.get_cmap("tab10")(c % 10), label=f"cluster {c}")
    ax[2].set(title=f"embedding: highlighted = {kind}", xticks=[], yticks=[]); ax[2].legend(markerscale=6); fig.tight_layout(); fig.savefig(out / "sweep.png", dpi=110); plt.close(fig)


def _bouts(lab: np.ndarray, c: int) -> list[tuple[int, int]]:
    m = np.r_[False, lab == c, False]; d = np.diff(m.astype(int)); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(s, e) if b - a >= MIN_BOUT]


def clips(R: dict, root: Path, out: Path, flag: list[int], tag: str, j: dict) -> list[str]:
    cams = {s["path"]: s["cams"] for s in core.scan(root)}
    font = ImageFont.load_default(size=15)
    rec, written = R["recommended"], []
    sess = []                                                              # (group index, path, session, cam path)
    for gi, g in enumerate(R["groups"]):
        n = R["n"][gi]; lo = 0 if gi == 0 else R["n"][0]
        for r in R["paths"][lo:lo + n]:
            if cams.get(r):
                sess.append((gi, r, core.load(root, r), root / cams[r][sorted(cams[r])[0]], int(sorted(cams[r])[0])))
    missing = [g for gi, g in enumerate(R["groups"]) if not any(x[0] == gi for x in sess)]
    if missing and sess:
        core._job_log(j, f"no per-camera video for group {missing}: clips show the other group only")
    if not sess:
        core._job_log(j, "no per-camera video among the project sessions: no clips"); return written
    for c in flag:
        cand = []
        for gi, r, s, vp, cam in sess:
            lab = core.pick_level(s["super"], rec["level"])
            cand += [(b - a, gi, r, s, vp, cam, a, b) for a, b in _bouts(lab, c)]
        pick = []
        for gi in (0, 1):
            pick += sorted([x for x in cand if x[1] == gi], key=lambda x: -x[0])[:PER_GROUP]
        if not pick:
            core._job_log(j, f"cluster {c}: no bout of at least {MIN_BOUT} frames in a session with video"); continue
        wd, ht = map(int, subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(pick[0][4])],
                                         capture_output=True, text=True, check=True).stdout.strip().split("x"))
        path = out / f"cluster_{c}_{tag}.mp4"
        enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{wd}x{ht}", "-r", "20", "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        for ln, gi, r, s, vp, cam, a, b in pick:
            mid = (a + b) // 2; st = max(0, min(a, mid - CLIP // 2)); n = min(CLIP, b - st)
            raw = subprocess.run([FF, "-v", "error", "-ss", f"{st / 20:.3f}", "-i", str(vp), "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            fr = np.frombuffer(raw, np.uint8).reshape(-1, ht, wd, 3); kp = core.kp_cell(s, cam)
            for i, f in enumerate(fr):
                im = Image.fromarray(f.copy()); d = ImageDraw.Draw(im); P = kp[st + i]
                for u, v in s["edges"]:
                    if P[u][2] > .25 and P[v][2] > .25: d.line([P[u][0] * wd, P[u][1] * ht, P[v][0] * wd, P[v][1] * ht], fill=COL[gi], width=2)
                for p in P:
                    if p[2] > .25: d.ellipse([p[0] * wd - 3, p[1] * ht - 3, p[0] * wd + 3, p[1] * ht + 3], fill=COL[gi])
                d.rectangle([0, 0, wd, 22], fill=(0, 0, 0)); d.text((4, 3), f"cluster {c} ({tag}) | {R['groups'][gi]} | {Path(r).stem[:18]} | frame {st + i}", fill=COL[gi], font=font)
                enc.stdin.write(np.asarray(im).tobytes())
        enc.stdin.close(); enc.wait(); written.append(path.name); core._job_log(j, f"clip {path.name}: {len(pick)} bouts")
    return written


def start(root: Path, name: str, criterion: str = "omnibus") -> str:
    return model._job(root, _run, name, criterion)


def _run(root: Path, j: dict, name: str, criterion: str = "omnibus") -> None:
    j.update(stage="sweep", pct=5)
    R = sweep.project_sweep(root, core.load_project(root, name), criterion); rec = R["recommended"]
    out = root / ".subtle_results" / name / f"{time.strftime('%y%m%d_%H%M%S')}_n{rec['k']}"; out.mkdir(parents=True)
    tables(R, out); j.update(stage="figure", pct=30); figure(R, root, out)
    lv = next(o for o in R["levels"] if o["level"] == rec["level"])
    flag, tag = (rec["significant"], "significant") if rec["significant"] else (np.array(lv["ids"])[np.argsort(-np.abs(lv["t"]))[:TOP_EXPLORATORY]].tolist(), "exploratory_not_significant")
    j.update(stage=f"clips for clusters {flag}", pct=45); files = clips(R, root, out, [int(c) for c in flag], tag, j)
    summary = json.loads((out / "summary.json").read_text()); summary["files"] = sorted(p.name for p in out.iterdir()); summary["clips"] = files
    (out / "summary.json").write_text(json.dumps(summary, indent=1)); (root / ".subtle_results" / name / "latest.txt").write_text(str(out))
    j.update(stage="done", pct=100); core._job_log(j, f"saved {out}")
