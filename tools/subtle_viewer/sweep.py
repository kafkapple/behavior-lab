"""Choose the supercluster level (number of clusters) by sweeping every level, with the selection itself corrected.
Statistic: |Welch t| of each cluster's per-animal share, group A vs B. Null: the same statistic over all splits of the animals into groups of
the observed sizes (exact when <= CAP splits). Taking the best of many levels x clusters inflates false positives, so every p is a
max-T p: the share of null splits whose maximum over ALL levels and clusters reaches the observed value. No web code."""
from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

import core
import stats

CAP = 20_000


def shares(labs: list[np.ndarray], ids: list[int]) -> np.ndarray:
    """(animals, clusters) percent of each animal's labeled frames."""
    return np.array([[100.0 * (l == c).sum() / len(l) for c in ids] for l in labs])


def welch_t(S: np.ndarray, a: list[int], b: list[int]) -> np.ndarray:
    x, y = S[a], S[b]
    den = np.sqrt(x.var(0, ddof=1) / len(a) + y.var(0, ddof=1) / len(b))
    d = x.mean(0) - y.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(den > 0, d / np.where(den > 0, den, 1), np.where(d == 0, 0.0, np.sign(d) * 1e6))
    return t


def splits(n: int, na: int, seed: int = 0) -> tuple[list[tuple[int, ...]], bool]:
    if comb(n, na) <= CAP:
        return list(combinations(range(n), na)), True
    rng = np.random.default_rng(seed)
    return [tuple(sorted(rng.permutation(n)[:na])) for _ in range(CAP)], False


def sweep_levels(levels: dict[int, list[np.ndarray]], groups: list[list[int]], names: list[str]) -> dict:
    """levels: level index -> per-animal label arrays. groups: animal indices of group A and B (first two groups)."""
    A, B = groups[0], groups[1]
    animals = A + B
    n, na = len(animals), len(A)
    sp, exact = splits(n, na)
    obs_split = tuple(range(na))                                  # animals reordered so that A comes first
    out, null_t, null_d = [], {}, {}
    for j, labs in levels.items():
        labs = [labs[i] for i in animals]
        ids = sorted({int(c) for l in labs for c in np.unique(l)})
        if len(ids) < 2:
            continue                                              # one cluster has no variance to test
        S = shares(labs, ids)
        t_obs, d_obs = welch_t(S, list(range(na)), list(range(na, n))), S[:na].mean(0) - S[na:].mean(0)
        T = np.array([np.abs(welch_t(S, list(s), [i for i in range(n) if i not in s])) for s in sp])
        D = np.array([np.abs(S[list(s)].mean(0) - np.delete(S, list(s), 0).mean(0)) for s in sp])
        null_t[j], null_d[j] = T, D
        out.append({"level": j, "k": len(ids), "ids": ids, "t": t_obs.tolist(), "diff": d_obs.tolist(), "means": {names[0]: S[:na].mean(0).tolist(), names[1]: S[na:].mean(0).tolist()},
                    "values": {names[0]: S[:na].T.tolist(), names[1]: S[na:].T.tolist()},
                    "g": [stats.hedges_g(S[:na, c], S[na:, c]) for c in range(len(ids))],
                    "p_raw": [float((D[:, c] >= abs(d_obs[c]) - 1e-12).mean()) for c in range(len(ids))],
                    "max_abs_t": float(np.abs(t_obs).max())})
    if not out:
        raise ValueError("no level with at least 2 clusters")
    gmax = np.max([null_t[o["level"]].max(1) for o in out], axis=0)           # per split: maximum over every level and cluster
    obs_global = max(o["max_abs_t"] for o in out)
    thr_global = float(np.quantile(gmax, 0.95))
    for o in out:
        Tm = null_t[o["level"]].max(1)
        o["p_level"] = float((Tm >= o["max_abs_t"] - 1e-9).mean())          # best cluster within this level
        o["thr95"] = float(np.quantile(Tm, 0.95))
        o["score"] = o["max_abs_t"] / o["thr95"] if o["thr95"] > 0 else 0.0   # above 1 = passes this level's own 95 percent line
        o["p_adj"] = [float((gmax >= abs(t) - 1e-9).mean()) for t in o["t"]]   # selection-corrected, every level and cluster
    best = max(out, key=lambda o: (round(o["score"], 6), -o["k"]))
    rec = {"level": best["level"], "k": best["k"], "score": best["score"], "p_level": best["p_level"], "p_fw": float((gmax >= obs_global - 1e-9).mean()),
           "significant": [c for c, q in zip(best["ids"], best["p_adj"]) if q < 0.05],
           "exploratory": [c for c, q, r in zip(best["ids"], best["p_adj"], best["p_raw"]) if q >= 0.05 and r < 0.05]}
    return {"levels": out, "recommended": rec, "groups": names[:2], "n": [na, n - na], "splits": len(sp), "exact": exact, "global_thr95": thr_global,
            "min_possible_p": 1 / len(sp) if exact else None,
            "method": f"max-T over {len(out)} levels x clusters; {'exact' if exact else 'Monte Carlo'} {len(sp)} splits of {n} animals into {na}+{n - na}"}


def project_sweep(root: Path, proj: dict) -> dict:
    gn = [g for g, s in proj.get("groups", {}).items() if s]
    if len(gn) < 2:
        raise ValueError("need two non-empty groups")
    paths = [r for g in gn[:2] for r in proj["groups"][g]]
    S = [core.load(root, r) for r in paths]
    if any(s["super"] is None for s in S) or len({s["fit_id"] for s in S}) != 1 or S[0]["fit_id"] is None:
        raise ValueError("every session needs supercluster labels from one fit")
    nl = [len(s["super"]) for s in S]
    raw = [np.asarray(s["super"]) for s in S]
    if raw[0].ndim != 2:
        raise ValueError("the fit has one supercluster level only; run SUBTLE again")
    levels = {j: [r[:, j] for r in raw] for j in range(raw[0].shape[1])}
    idx, i = [], 0
    for g in gn[:2]:
        idx.append(list(range(i, i + len(proj["groups"][g])))); i += len(proj["groups"][g])
    res = sweep_levels(levels, idx, gn[:2])
    res.update({"fit_id": S[0]["fit_id"], "paths": paths, "frames": nl})
    return res
