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


def safe(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if root.resolve() not in p.parents and p != root.resolve():
        raise ValueError(f"path outside root: {rel}")
    return p


def stem_of(p: Path) -> str:
    return STEM_SUFFIX.sub("", p.name.removesuffix(".gz").removesuffix(".json"))


def scan(root: Path) -> list[dict]:
    videos = {}
    for v in root.rglob("*.mp4"):
        if not any(s.startswith(".") for s in v.relative_to(root).parts):
            videos.setdefault(v.stem, v)
    out = []
    for p in sorted(root.rglob("*.json.gz")):
        rel = p.relative_to(root)
        if any(s.startswith(".") for s in rel.parts):
            continue
        v = videos.get(stem_of(p))
        side = p.with_name(p.name.removesuffix(".json.gz") + SIDE)
        out.append({"path": str(rel), "mb": round(p.stat().st_size / 1e6, 2), "stem": stem_of(p),
                    "video": str(v.relative_to(root)) if v else None, "subtle": side.exists()})
    log.info("scan %s: %d sessions, %d videos", root, len(out), len(videos))
    return out


def _side(p: Path) -> Path:
    return p.with_name(p.name.removesuffix(".json.gz") + SIDE)


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
         "xyz": xyz, "sub": None, "super": None, "fit_id": None}
    emb = d.get("subtle") or {}
    s["sub"], s["super"] = emb.get("subcluster"), emb.get("supercluster")
    if _side(p).exists():
        sd = json.loads(_side(p).read_text())
        s["sub"], s["super"], s["fit_id"] = sd.get("subcluster"), sd.get("supercluster"), sd.get("fit_id")
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
        for rel, a, b, n in zip(j["paths"], sub_parts, sup_parts, lengths):
            _side(safe(root, rel)).write_text(json.dumps(
                {"subcluster": a, "supercluster": b, "n_frames": n, "fit_id": fit_id, "fit_sessions": j["paths"]}))
        _CACHE.clear()  # sidecars changed
        j.update(stage="done", pct=100)
        _job_log(j, f"fit {fit_id}: clusters {res.n_clusters}, frames {sum(lengths)}, {time.time() - j['t0']:.0f}s")
    except Exception as e:  # report to UI; do not kill the server
        j["error"] = f"{type(e).__name__}: {e}"
        _job_log(j, "ERROR " + j["error"])
    finally:
        j["done"] = True
