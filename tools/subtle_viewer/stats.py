"""Group statistics on cluster shares (unit = animal = one session) and user cluster names. No web code."""
from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

import core

STARS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def stars(p: float) -> str:
    return next((s for t, s in STARS if p < t), "ns")


def perm_p(a: np.ndarray, b: np.ndarray, seed: int = 0, cap: int = 200_000) -> tuple[float, str]:
    """Two-sided permutation p of |mean difference|; exact when the number of splits <= cap, else Monte Carlo (20,000)."""
    x, n, k = np.r_[a, b], len(a) + len(b), len(a)
    obs = abs(a.mean() - b.mean())
    if comb(n, k) <= cap:
        diffs = np.array([abs(x[list(i)].mean() - np.delete(x, list(i)).mean()) for i in combinations(range(n), k)])
        return float((diffs >= obs - 1e-12).mean()), "exact permutation"
    rng = np.random.default_rng(seed)
    d = [abs(x[p[:k]].mean() - x[p[k:]].mean()) for p in (rng.permutation(n) for _ in range(20_000))]
    return float((np.array(d) >= obs - 1e-12).mean()), "Monte Carlo permutation (20,000)"


def hedges_g(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return None if sp == 0 else float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def bh(p: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p."""
    o = np.argsort(p)
    adj = np.minimum.accumulate((np.array(p)[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[o] = np.minimum(adj, 1.0)
    return out.tolist()


def animal_shares(root: Path, sessions: list[str], key: str, sl: int | None) -> dict[str, dict[int, float]]:
    out = {}
    for rel in sessions:
        lab = core.load(root, rel)[key]
        if lab is None:
            raise ValueError(f"no labels in {rel}: run SUBTLE first")
        lab = core.pick_level(lab, sl)
        out[rel] = {int(c): float(v) * 100 / len(lab) for c, v in zip(*np.unique(lab, return_counts=True))}
    return out


def group_stats(root: Path, proj: dict, level: str = "super", sl: int | None = None, correct: bool = True) -> dict:
    key = "super" if level == "super" else "sub"
    groups = {g: s for g, s in proj.get("groups", {}).items() if s}
    if len(groups) < 2:
        raise ValueError("need at least two non-empty groups")
    fits = {core.load(root, r)["fit_id"] for s in groups.values() for r in s}
    shares = {g: animal_shares(root, s, key, sl) for g, s in groups.items()}
    names = list(groups)
    pairs = []
    for ga, gb in combinations(names, 2):
        cl = sorted({c for g in (ga, gb) for sh in shares[g].values() for c in sh})
        rows = []
        for c in cl:
            a = np.array([sh.get(c, 0.0) for sh in shares[ga].values()])
            b = np.array([sh.get(c, 0.0) for sh in shares[gb].values()])
            p, method = perm_p(a, b)
            rows.append({"cluster": c, "mean": {ga: float(a.mean()), gb: float(b.mean())}, "values": {ga: a.tolist(), gb: b.tolist()},
                         "diff": float(a.mean() - b.mean()), "g": hedges_g(a, b), "p": p})
        padj = bh([r["p"] for r in rows]) if correct else [r["p"] for r in rows]
        for r, q in zip(rows, padj):
            r["p_adj"], r["stars"] = q, stars(q)
        na, nb = len(shares[ga]), len(shares[gb])
        pairs.append({"groups": [ga, gb], "n": {ga: na, gb: nb}, "min_p": 2 / comb(na + nb, na), "method": method, "tests": rows,
                      "correction": "Benjamini-Hochberg over the clusters of this level" if correct else "none"})
    return {"pairs": pairs, "one_fit": len(fits) == 1 and None not in fits, "unit": "animal (one session = one animal assumed)"}


def names_path(root: Path) -> Path:
    return core.project_dir(root) / "cluster_names.json"


def names_all(root: Path) -> dict:
    p = names_path(root)
    return json.loads(p.read_text()) if p.exists() else {}


def names_set(root: Path, key: str, name: str) -> dict:
    """key = '<fit_id>|<level>|<id>'; an empty name removes the entry."""
    if len(key) > 80 or len(name) > 60 or "\n" in name:
        raise ValueError("bad cluster name")
    d = names_all(root)
    d.pop(key, None) if not name.strip() else d.__setitem__(key, name.strip())
    names_path(root).write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return d
