"""SUBTLE Lite Viewer core: scan, load, projects, frequency, SUBTLE job.

No web code here, so every function can be asserted in selftest.py.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

log = logging.getLogger("subtle_viewer")
SIDE = ".subtle.json"  # sidecar suffix; the original json.gz is never modified
STEM_SUFFIX = re.compile(r"(_det|_sleap.*|_tailless.*)$")
NAME_OK = re.compile(r"^[\w\-. ]{1,64}$")
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()  # LRU of parsed sessions
CACHE_MAX = 2
JOBS: "OrderedDict[str, dict]" = OrderedDict()
JOBS_MAX = 20
# AVATAR rig (configs/rig/avatar.yaml): composite 3600x2000, camera cell = (x0, y0, w, h) in composite px.
# ponytail: cam_3 height 1200 is from the yaml comment, not measured; the page warns when a video's aspect differs.
COMPOSITE = (3600, 2000)
CELLS = {0: (0, 0, 1200, 1000), 1: (1200, 0, 1200, 1000), 2: (2400, 0, 1200, 1200),
         3: (0, 1000, 1200, 1000), 4: (1200, 1000, 1200, 1000)}


def safe(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if root.resolve() not in p.parents and p != root.resolve():
        raise ValueError(f"path outside root: {rel}")
    return p


def stem_of(p: Path) -> str:
    return STEM_SUFFIX.sub("", p.name.removesuffix(".gz").removesuffix(".json"))


def _token(stem: str) -> str | None:
    m = re.match(r"([A-Za-z]+\d+)", stem)
    return m.group(1) if m else None


def scan(root: Path) -> list[dict]:
    vids = [v for v in root.rglob("*.mp4") if not any(s.startswith(".") for s in v.relative_to(root).parts)]
    videos = {}
    for v in vids:
        videos.setdefault(v.stem, v)
    out = []
    for p in sorted(root.rglob("*.json.gz")):
        rel = p.relative_to(root)
        if any(s.startswith(".") for s in rel.parts):
            continue
        v = videos.get(stem_of(p))
        tok, cams = _token(stem_of(p)), {}
        if tok:  # per-camera videos: name holds the animal token as a whole word and `cam_N`
            pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(tok)}(?!\d)")
            for x in vids:
                m = re.search(r"cam_(\d)", x.stem)
                if m and pat.search(x.stem) and 1 <= int(m.group(1)) <= 5:
                    cams[int(m.group(1)) - 1] = str(x.relative_to(root))
        side = p.with_name(p.name.removesuffix(".json.gz") + SIDE)
        out.append({"path": str(rel), "mb": round(p.stat().st_size / 1e6, 2), "stem": stem_of(p),
                    "video": str(v.relative_to(root)) if v else None, "cams": cams, "subtle": side.exists()})
    log.info("scan %s: %d sessions, %d videos", root, len(out), len(vids))
    return out


def _side(p: Path) -> Path:
    return p.with_name(p.name.removesuffix(".json.gz") + SIDE)


def _kp_array(d: dict, n: int) -> np.ndarray | None:
    """keypoint[frame][camera][node] = [x, y, conf] (x, y relative to the composite) -> (n, 5, nodes, 3) float32."""
    kp = d.get("keypoint")
    if not kp or n == 0:
        return None
    nodes = len(d.get("node", {}).get("id", [])) or 11
    out = np.zeros((n, 5, nodes, 3), np.float32)
    for f, cams in kp.items():
        i = int(f)
        if i < n:
            for c, pts in cams.items():
                for k, v in pts.items():
                    out[i, int(c), int(k)] = v
    return out


def kp_cell(s: dict, cam: int) -> np.ndarray:
    """2D keypoints of one camera as (T, nodes, 3): x, y in 0-1 of that camera's cell, conf unchanged."""
    x0, y0, w, h = CELLS[cam]
    a = s["kp"][:, cam].copy()
    a[..., 0] = (a[..., 0] * COMPOSITE[0] - x0) / w
    a[..., 1] = (a[..., 1] * COMPOSITE[1] - y0) / h
    return a


def load(root: Path, rel: str) -> dict:
    p = safe(root, rel)
    key = (str(p), p.stat().st_mtime)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    t0 = time.time()
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    sk = d.get("skeleton3d") or {}
    frames = sorted(sk, key=int)
    xyz = np.array([sk[k] for k in frames], dtype=np.float32) if frames else np.zeros((0, 0, 3), np.float32)
    s = {"nodes": d.get("node", {}).get("id", []), "edges": d.get("node", {}).get("line", []),
         "xyz": xyz, "sub": None, "super": None, "fit_id": None, "emb": None,
         "kp": _kp_array(d, len(frames))}
    emb = d.get("subtle") or {}
    s["sub"], s["super"] = emb.get("subcluster"), emb.get("supercluster")
    if _side(p).exists():
        sd = json.loads(_side(p).read_text())
        s["sub"], s["super"], s["fit_id"] = sd.get("subcluster"), sd.get("supercluster"), sd.get("fit_id")
        s["emb"] = sd.get("embedding")
    del d  # raw dict can be 40 MB+; keep only arrays
    _CACHE[key] = s
    while len(_CACHE) > CACHE_MAX:
        _CACHE.popitem(last=False)
    log.info("load %s: %s frames, %.1fs", rel, xyz.shape[0], time.time() - t0)
    return s


def project_dir(root: Path) -> Path:
    d = root / ".subtle_projects"
    d.mkdir(exist_ok=True)
    return d


def save_project(root: Path, proj: dict) -> str:
    name = proj.get("name", "")
    if not NAME_OK.match(name):
        raise ValueError("name: 1-64 chars of letters, digits, space, - _ .")
    (project_dir(root) / f"{name}.project.json").write_text(json.dumps(proj, ensure_ascii=False, indent=1))
    log.info("save project %s (%d groups)", name, len(proj.get("groups", {})))
    return name


def load_project(root: Path, name: str) -> dict:
    if not NAME_OK.match(name):
        raise ValueError("bad name")
    return json.loads((project_dir(root) / f"{name}.project.json").read_text())


def list_projects(root: Path) -> list[str]:
    return sorted(p.name.removesuffix(".project.json") for p in project_dir(root).glob("*.project.json"))


def frequency(root: Path, proj: dict, level: str = "super") -> dict:
    """Per-group cluster pct. ok=False when a group's pct sum is not 100 or a session has no labels."""
    key, res, fits = ("super" if level == "super" else "sub"), {}, set()
    for g, sessions in proj.get("groups", {}).items():
        cnt, missing = Counter(), []
        for rel in sessions:
            ses = load(root, rel)
            lab = ses[key]
            fits.add(ses["fit_id"])
            if lab is None:
                missing.append(rel)
                continue
            cnt.update(int(x) for x in np.asarray(lab).reshape(len(lab), -1)[:, -1])
        n = sum(cnt.values())
        pct = {str(c): 100.0 * v / n for c, v in sorted(cnt.items())} if n else {}
        total = sum(pct.values())
        res[g] = {"n": n, "pct": pct, "sum": total, "missing": missing,
                  "ok": bool(n) and abs(total - 100.0) < 1e-6 and not missing}
    same_fit = len(fits) == 1 and None not in fits  # ids are comparable only inside one joint fit
    for r in res.values():
        r["same_fit"], r["ok"] = same_fit, r["ok"] and same_fit
    return res


def _job_log(j: dict, msg: str) -> None:
    j["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
    log.info("job %s: %s", j["id"], msg)


def split_by_lengths(labels, lengths: list[int]) -> list[list]:
    """Cut the concatenated joint-fit labels back into one list per session; length must match exactly."""
    assert len(labels) == sum(lengths), f"label count {len(labels)} != frames {sum(lengths)}"
    cuts = np.cumsum([0, *lengths])
    return [np.asarray(labels[a:b]).tolist() for a, b in zip(cuts[:-1], cuts[1:])]


def start_job(root: Path, paths: list[str], max_frames: int = 0) -> str:
    """One SUBTLE fit over ALL given sessions, so cluster ids mean the same thing in every session."""
    j = {"id": uuid.uuid4().hex[:8], "paths": paths, "stage": "queued", "pct": 0, "t0": time.time(),
         "done": False, "error": None, "log": []}
    JOBS[j["id"]] = j
    while len(JOBS) > JOBS_MAX:
        JOBS.popitem(last=False)
    threading.Thread(target=_run_job, args=(root, j, max_frames), daemon=True).start()
    return j["id"]


def _run_job(root: Path, j: dict, max_frames: int) -> None:
    try:
        j.update(stage="load", pct=10)
        seqs = []
        for rel in j["paths"]:
            xyz = load(root, rel)["xyz"]
            xyz = xyz[:max_frames] if max_frames else xyz
            _job_log(j, f"input {rel} {xyz.shape}, NaN {int(np.isnan(xyz).sum())}")
            seqs.append(np.nan_to_num(xyz))  # ponytail: zero-fill; interpolate if NaN share is large
        lengths = [len(s) for s in seqs]
        j.update(stage=f"joint fit on {len(seqs)} sessions (UMAP + clustering, isolated subprocess)", pct=30)
        from behavior_lab.models.discovery.subtle_wrapper import SUBTLE
        res = SUBTLE(fps=20).fit_predict(seqs, use_superclusters=True, isolate=True)
        j.update(stage="write", pct=90)
        sub = res.metadata.get("subclusters")
        sup_parts = split_by_lengths(np.asarray(res.labels), lengths)
        sub_parts = split_by_lengths(np.asarray(sub), lengths) if sub is not None else [None] * len(seqs)
        fit_id = uuid.uuid4().hex[:8]  # same id on every session of this fit; frequency() requires one id per project
        emb = getattr(res, "embeddings", None)
        emb_parts = split_by_lengths(np.round(np.asarray(emb), 3), lengths) if emb is not None else [None] * len(seqs)
        for rel, a, b, e, n in zip(j["paths"], sub_parts, sup_parts, emb_parts, lengths):
            _side(safe(root, rel)).write_text(json.dumps(
                {"subcluster": a, "supercluster": b, "embedding": e, "n_frames": n, "fit_id": fit_id,
                 "fit_sessions": j["paths"]}))
        _CACHE.clear()  # sidecars changed
        j.update(stage="done", pct=100)
        _job_log(j, f"fit {fit_id}: clusters {res.n_clusters}, frames {sum(lengths)}, {time.time() - j['t0']:.0f}s")
    except Exception as e:  # report to UI; do not kill the server
        j["error"] = f"{type(e).__name__}: {e}"
        _job_log(j, "ERROR " + j["error"])
    finally:
        j["done"] = True
