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
    """All splits of n animals into na + (n - na); the observed split (animals 0..na-1) is always first."""
    obs = tuple(range(na))
    if comb(n, na) <= CAP:
        return [obs] + [s for s in combinations(range(n), na) if s != obs], True
    rng = np.random.default_rng(seed)
    return [obs] + [tuple(sorted(rng.permutation(n)[:na])) for _ in range(CAP - 1)], False


def pseudo_f(X: np.ndarray, a: list[int], b: list[int]) -> float:
    """PERMANOVA pseudo-F (Euclidean distance) of Hellinger-transformed shares: between-group over within-group sum of squares."""
    ss = lambda Z: float(((Z - Z.mean(0)) ** 2).sum())  # noqa: E731
    within = ss(X[a]) + ss(X[b])
    return (ss(X) - within) / (within / (len(a) + len(b) - 2)) if within > 0 else 0.0


CRITERIA = {"omnibus": "pseudo-F of all cluster shares together (Hellinger)", "maxt": "largest |t| of any cluster",
            "nsig": "number of clusters with raw permutation p < .05", "meanp": "mean raw permutation p over clusters (lower is better)"}


def sweep_levels(levels: dict[int, list[np.ndarray]], groups: list[list[int]], names: list[str], criterion: str = "omnibus") -> dict:
    """levels: level index -> per-animal label arrays. groups: animal indices of group A and B (first two groups).
    Every criterion is a statistic with its own permutation null; levels are compared on the z-score against that null,
    and the maximum over levels is again referred to the null (selection-corrected p_fw)."""
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {sorted(CRITERIA)}")
    A, B = groups[0], groups[1]
    animals = A + B
    n, na = len(animals), len(A)
    sp, exact = splits(n, na)
    S = len(sp)
    out, stat = [], {}                                              # stat[level][criterion] = array over splits (larger = stronger evidence)
    for j, labs in levels.items():
        labs = [labs[i] for i in animals]
        ids = sorted({int(c) for l in labs for c in np.unique(l)})
        if len(ids) < 2:
            continue                                              # one cluster has no variance to test
        Sh = shares(labs, ids)
        H = np.sqrt(Sh / 100.0)
        T = np.array([np.abs(welch_t(Sh, list(s), [i for i in range(n) if i not in s])) for s in sp])             # (splits, clusters)
        D = np.array([np.abs(Sh[list(s)].mean(0) - np.delete(Sh, list(s), 0).mean(0)) for s in sp])
        F = np.array([pseudo_f(H, list(s), [i for i in range(n) if i not in s]) for s in sp])
        P = np.empty_like(D)                                       # raw permutation p of every split's cluster difference, by rank in that cluster's null
        for c in range(D.shape[1]):
            srt = np.sort(D[:, c]); P[:, c] = (S - np.searchsorted(srt, D[:, c] - 1e-12, side="left")) / S
        stat[j] = {"omnibus": F, "maxt": T.max(1), "nsig": (P < 0.05).sum(1).astype(float), "meanp": -P.mean(1)}
        t_obs, d_obs = T[0] * np.sign(welch_t(Sh, list(range(na)), list(range(na, n))) + 1e-300), Sh[:na].mean(0) - Sh[na:].mean(0)
        out.append({"level": j, "k": len(ids), "ids": ids, "t": t_obs.tolist(), "diff": d_obs.tolist(), "means": {names[0]: Sh[:na].mean(0).tolist(), names[1]: Sh[na:].mean(0).tolist()},
                    "values": {names[0]: Sh[:na].T.tolist(), names[1]: Sh[na:].T.tolist()}, "g": [stats.hedges_g(Sh[:na, c], Sh[na:, c]) for c in range(len(ids))],
                    "p_raw": P[0].tolist(), "max_abs_t": float(T[0].max()), "_T": T})
    if not out:
        raise ValueError("no level with at least 2 clusters")
    z = {}                                                          # z[criterion][level] = null-standardized statistic per split
    for cr in CRITERIA:
        z[cr] = {o["level"]: (stat[o["level"]][cr] - stat[o["level"]][cr].mean()) / (stat[o["level"]][cr].std() or 1.0) for o in out}
    gz = {cr: np.max([z[cr][o["level"]] for o in out], axis=0) for cr in CRITERIA}               # per split: best level
    gmax = np.max([o["_T"].max(1) for o in out], axis=0)           # per split: largest |t| over every level and cluster (cluster-level flags)
    for o in out:
        T = o.pop("_T"); Tm = T.max(1)
        o["p_adj"] = [float((gmax >= abs(t) - 1e-9).mean()) for t in o["t"]]
        o["p_level"] = float((Tm >= o["max_abs_t"] - 1e-9).mean()); o["thr95"] = float(np.quantile(Tm, 0.95))
        o["score"] = o["max_abs_t"] / o["thr95"] if o["thr95"] > 0 else 0.0
        o["criteria"] = {cr: {"obs": float(stat[o["level"]][cr][0]), "z": float(z[cr][o["level"]][0]), "p_level": float((stat[o["level"]][cr] >= stat[o["level"]][cr][0] - 1e-12).mean())} for cr in CRITERIA}
    by = {}
    for cr in CRITERIA:                                             # recommendation per criterion; ties go to fewer clusters
        best = max(out, key=lambda o: (round(o["criteria"][cr]["z"], 6), -o["k"]))
        by[cr] = {"level": best["level"], "k": best["k"], "z": best["criteria"][cr]["z"], "p_fw": float((gz[cr] >= gz[cr][0] - 1e-9).mean())}
    best = next(o for o in out if o["level"] == by[criterion]["level"])
    rec = {**by[criterion], "criterion": criterion, "score": best["score"], "p_level": best["criteria"][criterion]["p_level"],
           "significant": [c for c, q in zip(best["ids"], best["p_adj"]) if q < 0.05],
           "exploratory": [c for c, q, r in zip(best["ids"], best["p_adj"], best["p_raw"]) if q >= 0.05 and r < 0.05]}
    return {"levels": out, "recommended": rec, "by_criterion": by, "criteria": CRITERIA, "groups": names[:2], "n": [na, n - na], "splits": S, "exact": exact,
            "global_thr95": float(np.quantile(gmax, 0.95)), "min_possible_p": 1 / S if exact else None,
            "method": f"{CRITERIA[criterion]}; z against its own null, best level vs the null of the best level; {'exact' if exact else 'Monte Carlo'} {S} splits of {n} animals into {na}+{n - na}"}


def project_sweep(root: Path, proj: dict, criterion: str = "omnibus") -> dict:
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
    res = sweep_levels(levels, idx, gn[:2], criterion)
    res.update({"fit_id": S[0]["fit_id"], "paths": paths, "frames": nl})
    return res
