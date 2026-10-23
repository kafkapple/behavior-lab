"""Train a SUBTLE model once, save it, and map any session with it (all frames). Jobs share core.JOBS."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

import core

REPO = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "LOKY_MAX_CPU_COUNT": "1", "OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
NAME_OK = re.compile(r"^[\w\-]{1,48}$")


def mdir(root: Path, name: str) -> Path:
    if not NAME_OK.match(name):
        raise ValueError("model name: 1-48 letters, digits, - _")
    d = root / ".subtle_models" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_models(root: Path) -> list[dict]:
    out = []
    for p in sorted((root / ".subtle_models").glob("*/model.json")) if (root / ".subtle_models").exists() else []:
        out.append(json.loads(p.read_text()))
    return out


def _run(args: list[str], timeout: int = 3600) -> None:
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "subtle_proc.py"), *args], cwd=REPO, env=ENV,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"subprocess exit {r.returncode}: {(r.stderr or '')[-400:]}")


def _job(root: Path, fn, *a) -> str:
    j = {"id": uuid.uuid4().hex[:8], "paths": [], "stage": "queued", "pct": 0, "t0": time.time(), "done": False, "error": None, "log": []}
    core.JOBS[j["id"]] = j
    while len(core.JOBS) > core.JOBS_MAX:
        core.JOBS.popitem(last=False)

    def run():
        try:
            fn(root, j, *a)
        except Exception as e:  # shown in the page; the server stays up
            j["error"] = f"{type(e).__name__}: {e}"
            core._job_log(j, "ERROR " + j["error"])
        finally:
            j["done"] = True
    threading.Thread(target=run, daemon=True).start()
    return j["id"]


def _seqs(root: Path, paths: list[str], max_frames: int, j: dict) -> list[np.ndarray]:
    out = []
    for rel in paths:
        x = core.load(root, rel)["xyz"]
        x = x[:max_frames] if max_frames else x
        core._job_log(j, f"input {rel} {x.shape}, NaN {int(np.isnan(x).sum())}")
        out.append(np.nan_to_num(x))
    return out


def _write(root, rel, lab: dict, extra: dict) -> None:
    core._side(core.safe(root, rel)).write_text(json.dumps({**lab, **extra}))


def start_train(root: Path, name: str, paths: list[str], max_frames: int, seed: int | None) -> str:
    mdir(root, name)
    return _job(root, _train, name, paths, max_frames, seed)


def _train(root: Path, j: dict, name: str, paths: list[str], max_frames: int, seed: int | None) -> None:
    j.update(stage="load", pct=5)
    seqs = _seqs(root, paths, max_frames, j)
    lengths = [len(s) for s in seqs]
    d = mdir(root, name)
    with tempfile.TemporaryDirectory() as t:
        np.savez(Path(t) / "in.npz", keypoints=np.concatenate(seqs), lengths=np.array(lengths))
        j.update(stage=f"train on {len(seqs)} sessions (UMAP, Leiden, DIB; isolated subprocess)", pct=20)
        _run(["train", f"{t}/in.npz", f"{t}/out.npz", str(d / "model.pkl"), "none" if seed is None else str(seed)])
        r = np.load(Path(t) / "out.npz")
        emb, sub, sup = r["embeddings"], r["subclusters"], r["superclusters"]
    j.update(stage="write", pct=90)
    mid = uuid.uuid4().hex[:8]
    for rel, a, b, e, n in zip(paths, core.split_by_lengths(sub, lengths), core.split_by_lengths(sup, lengths),
                               core.split_by_lengths(np.round(emb, 3), lengths), lengths):
        _write(root, rel, {"subcluster": a, "supercluster": b, "embedding": e}, {"n_frames": n, "fit_id": mid, "seed": seed, "model": name})
    levels = [len(np.unique(sup[:, k])) for k in range(sup.shape[1])] if sup.ndim == 2 else [len(np.unique(sup))]
    (d / "model.json").write_text(json.dumps({"id": mid, "name": name, "seed": seed, "train_paths": paths, "frames": lengths,
        "max_frames": max_frames, "n_sub": int(len(np.unique(sub))), "levels": levels, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    core._CACHE.clear()
    j.update(stage="done", pct=100)
    core._job_log(j, f"model {name} id {mid}: {len(np.unique(sub))} subclusters, levels {levels}, {time.time() - j['t0']:.0f}s")


def start_map(root: Path, name: str, paths: list[str]) -> str:
    return _job(root, _map, name, paths)


def _map(root: Path, j: dict, name: str, paths: list[str]) -> None:
    d = mdir(root, name)
    meta = json.loads((d / "model.json").read_text())
    for i, rel in enumerate(paths):
        j.update(stage=f"map {i + 1}/{len(paths)}: {rel}", pct=5 + int(90 * i / len(paths)))
        before = core.load(root, rel)
        old = before["sub"] if before["fit_id"] == meta["id"] else None  # labels from training, to check agreement
        x = [np.nan_to_num(before["xyz"])]
        with tempfile.TemporaryDirectory() as t:
            np.savez(Path(t) / "in.npz", keypoints=x[0], lengths=np.array([len(x[0])]))
            _run(["map", f"{t}/in.npz", f"{t}/out.npz", str(d / "model.pkl")])
            r = np.load(Path(t) / "out.npz")
        sub = np.asarray(r["subclusters"])
        _write(root, rel, {"subcluster": sub.tolist(), "supercluster": np.asarray(r["superclusters"]).tolist(),
                           "embedding": np.round(r["embeddings"], 3).tolist()},
               {"n_frames": len(sub), "fit_id": meta["id"], "seed": meta["seed"], "model": name, "mapped": True})
        if old is not None:
            n = min(len(old), len(sub))
            core._job_log(j, f"{rel}: mapped labels equal training labels on {100 * float((np.asarray(old)[:n] == sub[:n]).mean()):.1f}% of the {n} trained frames")
        core._CACHE.clear()
    j.update(stage="done", pct=100)
    core._job_log(j, f"mapped {len(paths)} sessions with {name}, {time.time() - j['t0']:.0f}s")
