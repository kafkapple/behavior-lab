"""DEK: kinematic features (public subtle.Kinematics) averaged per animal and cluster, compared between two groups.
Same steps as the Justin app (`avatar_preprocess` centering -> Kinematics -> per-cluster mean -> Welch t per cell), with the
node layout given by name, because our files have 11 nodes and the app's `avatar_configs` indexes 9. No web code."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from scipy import stats as sps

import core
import stats

MIN_FRAMES = 30   # an animal needs this many frames in a cluster for its mean to count
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
ANGLES = {        # vertex in the middle; all three nodes exist in the 11-node skeleton
    "head": ("nose", "neck", "tailstart"), "forelimbL": ("forelegL", "neck", "tailstart"), "forelimbR": ("forelegR", "neck", "tailstart"),
    "hindlimbL": ("hindlegL", "tailstart", "neck"), "hindlimbR": ("hindlegR", "tailstart", "neck"),
    "tailroot": ("neck", "tailstart", "tail"), "tailbend": ("tailstart", "tail", "tailend")}


def layout(nodes: list[str], edges: list[list[int]]) -> tuple[dict, dict, dict, list[str]]:
    """Kinematics config by node name, and the readable column names in the order Kinematics emits them."""
    nm = [n.removesuffix("1") for n in nodes]
    ix = {n: i for i, n in enumerate(nm)}
    nd = {n: i for i, n in enumerate(nm)}
    ed = {f"{nm[a]}-{nm[b]}": (a, b) for a, b in edges}
    an = {k: tuple(ix[x] for x in v) for k, v in ANGLES.items() if all(x in ix for x in v)}
    cols = [f"{t}{n}:{name}" for n in range(3) for t, d in (("V", nd), ("E", ed), ("A", an)) for name in d]
    return nd, ed, an, cols


def animal_table(s: dict, key: str, sl: int | None) -> tuple:
    """(cluster ids, mean feature matrix (clusters, features), frames per cluster, column names) for one session."""
    from subtle import Kinematics  # heavy import, kept out of the module top
    lab = core.pick_level(s[key], sl)
    x = s["xyz"][: len(lab)].astype(np.float64)
    x = x - x.mean((0, 1))                                   # avatar_preprocess
    nd, ed, an, cols = layout(s["nodes"], s["edges"])
    K = Kinematics(fs=20, nodes=nd, edges=ed, angles=an)(x).to_numpy()
    ids = np.unique(lab)
    return ids, np.array([K[lab == c].mean(0) for c in ids]), np.array([(lab == c).sum() for c in ids]), cols


def welch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch t and p per column; a, b = (animals, features). Same call as the app (scipy ttest_ind, equal_var=False)."""
    t, p = sps.ttest_ind(a, b, axis=0, equal_var=False)
    return t, p


def group_dek(root: Path, proj: dict, level: str = "super", sl: int | None = None, correct: bool = True, progress=None) -> dict:
    key = "super" if level == "super" else "sub"
    groups = {g: s for g, s in proj.get("groups", {}).items() if s}
    if len(groups) < 2:
        raise ValueError("need at least two non-empty groups")
    fits = {core.load(root, rel)["fit_id"] for sessions in groups.values() for rel in sessions}
    if len(fits) != 1 or None in fits:
        raise ValueError("DEK requires labels from one fit so cluster IDs are comparable")
    tabs = {}
    total = sum(map(len, groups.values()))
    done = 0
    for g, sess in groups.items():
        tabs[g] = []
        for r in sess:
            s = core.load(root, r)
            if s[key] is None:
                raise ValueError(f"no labels in {r}: run SUBTLE first")
            tabs[g].append(animal_table(s, key, sl))
            done += 1
            if progress:
                progress(done, total, r)
    cols = tabs[next(iter(tabs))][0][3]
    pairs = []
    for ga, gb in combinations(groups, 2):
        cl = sorted({int(c) for g in (ga, gb) for t in tabs[g] for c in t[0]})
        T, P, G, N, skipped = [], [], [], [], []
        for c in cl:
            def arr(g):  # animals that have this cluster with enough frames
                return np.array([t[1][list(t[0]).index(c)] for t in tabs[g] if c in t[0] and t[2][list(t[0]).index(c)] >= MIN_FRAMES])
            a, b = arr(ga), arr(gb)
            if len(a) < 2 or len(b) < 2:
                skipped.append({"cluster": c, "n": [len(a), len(b)]})
                T.append([None] * len(cols)); P.append([None] * len(cols)); G.append([None] * len(cols)); N.append([len(a), len(b)])
                continue
            t, p = welch(a, b)
            g = [stats.hedges_g(a[:, k], b[:, k]) for k in range(len(cols))]
            T.append([None if np.isnan(v) else float(v) for v in t]); P.append([None if np.isnan(v) else float(v) for v in p])
            G.append(g); N.append([len(a), len(b)])
        flat = [(i, j) for i, row in enumerate(P) for j, v in enumerate(row) if v is not None]
        adj = stats.bh([P[i][j] for i, j in flat]) if correct and flat else [P[i][j] for i, j in flat]
        PA = [[None] * len(cols) for _ in cl]
        for (i, j), q in zip(flat, adj):
            PA[i][j] = q
        pairs.append({"groups": [ga, gb], "clusters": cl, "n": N, "t": T, "p": P, "p_adj": PA, "g": G, "skipped": skipped, "tests": len(flat),
                      "correction": "Benjamini-Hochberg over every cluster x feature cell of this pair" if correct else "none"})
    return {"columns": cols, "pairs": pairs, "min_frames": MIN_FRAMES, "unit": "animal (one session = one animal assumed)"}


def start(root: Path, proj: dict, level: str = "super", sl: int | None = None, correct: bool = True) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        completed = [old for old, job in JOBS.items() if job["done"]]
        for old in completed[:-10]:
            del JOBS[old]
        JOBS[job_id] = {"stage": "Loading sessions", "pct": 0, "done": False, "error": None, "t0": time.time()}

    def run():
        try:
            def update(done: int, total: int, rel: str):
                with _LOCK:
                    JOBS[job_id].update(stage=f"Kinematics {done}/{total}: {Path(rel).name}", pct=round(90 * done / total))
            result = group_dek(root, proj, level, sl, correct, update)
            with _LOCK:
                JOBS[job_id].update(stage="Complete", pct=100, done=True, result=result)
        except Exception as exc:
            with _LOCK:
                JOBS[job_id].update(stage="Failed", done=True, error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=run, daemon=True).start()
    return job_id


def status(job_id: str) -> dict:
    with _LOCK:
        job = JOBS[job_id]
        return {k: v for k, v in job.items() if k != "result"} | {"elapsed": round(time.time() - job["t0"], 1)}


def result(job_id: str) -> dict:
    with _LOCK:
        job = JOBS[job_id]
        if not job["done"] or job["error"]:
            raise ValueError(job["error"] or "DEK is still running")
        return job["result"]
