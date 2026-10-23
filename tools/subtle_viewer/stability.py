"""Seed stability: fit the project sessions with several seeds (isolated subprocess, nothing written to the sidecars), then report
how much the labels agree (adjusted Rand index) and whether the group result repeats, at every supercluster level. No web code."""
from __future__ import annotations

import json
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

import core
import model
import stats


def result_path(root: Path, name: str) -> Path:
    return core.project_dir(root) / f"{name}.stability.json"


def level_result(lab: np.ndarray, lengths: list[int], groups: list[list[int]], names: list[str]) -> dict:
    """Group test on one label vector: min BH-adjusted p over clusters and how many clusters pass .05, for the first two groups."""
    per = [dict(zip(*np.unique(x, return_counts=True))) for x in core.split_by_lengths(lab, lengths)]
    tot = [len(x) for x in core.split_by_lengths(lab, lengths)]
    cl = sorted({int(c) for d in per for c in d})
    sh = lambda c, g: np.array([100.0 * per[i].get(c, 0) / tot[i] for i in g])  # noqa: E731
    p = [stats.perm_p(sh(c, groups[0]), sh(c, groups[1]))[0] for c in cl]
    q = stats.bh(p)
    return {"k": len(cl), "min_p_adj": float(min(q)), "n_sig": int(sum(v < 0.05 for v in q)), "groups": names[:2]}


def start(root: Path, name: str, max_frames: int, seeds: list[int]) -> str:
    return model._job(root, _run, name, max_frames, seeds)


def _run(root: Path, j: dict, name: str, max_frames: int, seeds: list[int]) -> None:
    proj = core.load_project(root, name)
    gnames = [g for g, s in proj["groups"].items() if s]
    if len(gnames) < 2:
        raise ValueError("need two non-empty groups")
    paths = [r for g in gnames for r in proj["groups"][g]]
    idx, i = [], 0
    for g in gnames:
        idx.append(list(range(i, i + len(proj["groups"][g])))); i += len(proj["groups"][g])
    j.update(stage="load", pct=3)
    seqs = model._seqs(root, paths, max_frames, j)
    lengths = [len(s) for s in seqs]
    runs = {}
    with tempfile.TemporaryDirectory() as t:
        np.savez(Path(t) / "in.npz", keypoints=np.concatenate(seqs), lengths=np.array(lengths))
        for n, sd in enumerate(seeds):
            j.update(stage=f"fit seed {sd} ({n + 1}/{len(seeds)})", pct=5 + int(90 * n / len(seeds)))
            model._run(["train", f"{t}/in.npz", f"{t}/o{sd}.npz", f"{t}/m{sd}.pkl", str(sd)])
            r = np.load(Path(t) / f"o{sd}.npz")
            runs[sd] = (r["subclusters"], np.atleast_2d(r["superclusters"].T).T)
            core._job_log(j, f"seed {sd}: {len(np.unique(runs[sd][0]))} subclusters, {runs[sd][1].shape[1]} levels")
    out = {"seeds": seeds, "max_frames": max_frames, "groups": gnames, "frames": lengths, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "per_seed": {}, "ari_sub": {}, "ari_super_default": {}, "note": "ARI compares label vectors over identical frames; ids differ between seeds, ARI does not need them to match"}
    for sd, (sub, sup) in runs.items():
        ks = [len(np.unique(sup[:, k])) for k in range(sup.shape[1])]
        out["per_seed"][str(sd)] = {"sub": level_result(sub, lengths, idx, gnames), "levels": [{"level": k, **level_result(sup[:, k], lengths, idx, gnames)} for k in range(sup.shape[1])],
                                    "default_level": int(np.argmax(ks))}
    for a, b in combinations(seeds, 2):
        out["ari_sub"][f"{a}-{b}"] = float(adjusted_rand_score(runs[a][0], runs[b][0]))
        da, db = (runs[s][1][:, out["per_seed"][str(s)]["default_level"]] for s in (a, b))
        out["ari_super_default"][f"{a}-{b}"] = float(adjusted_rand_score(da, db))
    # stable = every seed agrees on whether any cluster passes .05 at its default level
    flags = {sd: out["per_seed"][str(sd)]["levels"][out["per_seed"][str(sd)]["default_level"]]["n_sig"] > 0 for sd in seeds}
    out["star_pattern"] = "stable" if len(set(flags.values())) == 1 else "seed-dependent"
    out["star_flags"] = {str(k): v for k, v in flags.items()}
    result_path(root, name).write_text(json.dumps(out, indent=1))
    j.update(stage="done", pct=100)
    core._job_log(j, f"stability {name}: {out['star_pattern']}, ARI sub {out['ari_sub']}")


def node_sensitivity(root: Path, name: str, drop: list[str], max_frames: int = 3000, seed: int = 0) -> dict:
    """Refit SUBTLE without the named nodes (isolated, sidecars untouched) and rerun the level sweep on the new labels: does the group result
    depend on a node that the quality checks flag? Same seed and frames as the baseline fit."""
    proj = core.load_project(root, name)
    gn = [g for g, s in proj["groups"].items() if s][:2]
    paths = [r for g in gn for r in proj["groups"][g]]
    sess = [core.load(root, r) for r in paths]
    keep = [i for i, n in enumerate(sess[0]["nodes"]) if n.removesuffix("1") not in drop]
    seqs = [np.nan_to_num(s["xyz"][:max_frames][:, keep]) for s in sess]
    lengths = [len(x) for x in seqs]
    with tempfile.TemporaryDirectory() as t:
        np.savez(Path(t) / "in.npz", keypoints=np.concatenate(seqs), lengths=np.array(lengths))
        model._run(["train", f"{t}/in.npz", f"{t}/o.npz", f"{t}/m.pkl", str(seed)])
        sup = np.load(Path(t) / "o.npz")["superclusters"]
    sup = np.atleast_2d(sup.T).T
    per = [core.split_by_lengths(sup[:, k], lengths) for k in range(sup.shape[1])]
    levels = {k: [np.asarray(a) for a in per[k]] for k in range(sup.shape[1])}
    idx, i = [], 0
    for g in gn:
        idx.append(list(range(i, i + len(proj["groups"][g])))); i += len(proj["groups"][g])
    import sweep
    out = {"dropped": drop, "nodes_kept": [sess[0]["nodes"][i] for i in keep], "seed": seed, "frames": max_frames, "levels": sup.shape[1], "by_criterion": {}}
    for cr in sweep.CRITERIA:
        R = sweep.sweep_levels(levels, idx, gn, cr)
        out["by_criterion"][cr] = {**R["by_criterion"][cr], "significant": R["recommended"]["significant"]}
    return out
