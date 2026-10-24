"""Clustering behind one call (PRD P8): run(method, sessions, params) -> labels per session, embedding or None, fit_id.
Every method fits ALL given sessions jointly, so one fit_id = one cluster id space. Only methods that run here are listed.
ponytail: both entries are the SUBTLE wrapper (two clustering paths). B-SOiD is not listed: `hdbscan` is in neither venv, its labels are
10 fps bins (need alignment to frames) and HDBSCAN noise = -1; add it as a third entry when those are settled."""
from __future__ import annotations

import uuid

import numpy as np

import core
from behavior_lab.models.discovery.subtle_wrapper import CLUSTERINGS, SUBTLE, SUBTLEConfig

METHODS = tuple(f"subtle/{c}" for c in CLUSTERINGS)  # first = default (umap_phenograph)


def method_of(clustering: str) -> str:
    return f"subtle/{clustering}"


def _subtle(sessions: list[np.ndarray], params: dict, clustering: str) -> dict:
    cfg = SUBTLEConfig(fps=20, seed=params.get("seed"), clustering=clustering, timeout=params.get("timeout", 3600))  # consensus on all frames > 600 s
    res = SUBTLE(config=cfg).fit_predict(sessions, use_superclusters=True, isolate=True)
    sub, sup = res.metadata.get("subclusters"), res.metadata.get("superclusters")  # sup (T, levels): one column per DIB solution
    emb = getattr(res, "embeddings", None)
    return {"labels": np.asarray(sup if sup is not None else res.labels), "sub": None if sub is None else np.asarray(sub),
            "embedding": None if emb is None else np.round(np.asarray(emb), 3), "n_clusters": res.n_clusters}


def run(method: str, sessions: list[np.ndarray], params: dict | None = None) -> dict:
    """sessions = list of (T_i, nodes, 3) arrays without NaN. Returns per-session lists cut by frame count."""
    assert method in METHODS, f"method must be one of {METHODS}"
    lengths = [len(s) for s in sessions]
    r = _subtle(sessions, params or {}, method.split("/", 1)[1])
    cut = lambda a: None if a is None else core.split_by_lengths(a, lengths)  # noqa: E731  asserts total length
    out = {"method": method, "fit_id": uuid.uuid4().hex[:8], "n_clusters": r["n_clusters"], "lengths": lengths,
           "labels": cut(r["labels"]), "sub": cut(r["sub"]), "embedding": cut(r["embedding"])}
    check(out)
    return out


def check(out: dict) -> None:
    """PRD P8 per-method check: label count = frame count for every session; shares of every session sum to 100."""
    for n, lab in zip(out["lengths"], out["labels"]):
        assert len(lab) == n, f"{out['method']}: {len(lab)} labels for {n} frames"
        _, cnt = np.unique(core.pick_level(lab), return_counts=True)
        assert abs(sum(100.0 * c / n for c in cnt) - 100.0) < 1e-6
