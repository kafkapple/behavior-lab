"""Backend self-check with SYNTHETIC data in a temp dir (logic only, never a result).  run: python selftest.py"""
import gzip
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import core  # noqa: E402


def mk(d: Path, name: str, T: int, labels=None, fit_id="f1"):
    sk = {str(i): np.random.rand(11, 3).tolist() for i in range(T)}
    body = {"node": {"id": [f"n{i}" for i in range(11)], "line": [[i, i + 1] for i in range(10)]}, "skeleton3d": sk}
    p = d / f"{name}_sleap_1423.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        json.dump(body, f)
    if labels is not None:
        (d / f"{name}_sleap_1423.subtle.json").write_text(
            json.dumps({"subcluster": labels, "supercluster": labels, "fit_id": fit_id}))
    return p


def main():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        mk(root / "a" / "deep", "A1", 30, [0, 1, 2] * 10)
        mk(root / "b", "B1", 7, [0, 0, 1, 1, 1, 2, 2])
        mk(root, "C1", 5)  # no labels
        (root / "b" / "B1.mp4").write_bytes(b"x")
        s = {x["stem"]: x for x in core.scan(root)}
        assert set(s) == {"A1", "B1", "C1"}, s  # subfolders found
        assert s["B1"]["video"] == "b/B1.mp4" and s["A1"]["video"] is None  # stem pairing, `_sleap_1423` stripped
        assert s["A1"]["subtle"] and not s["C1"]["subtle"]
        ses = core.load(root, s["A1"]["path"])
        assert ses["xyz"].shape == (30, 11, 3) and len(ses["edges"]) == 10
        try:
            core.safe(root, "../etc/passwd")
            raise SystemExit("path traversal not blocked")
        except ValueError:
            pass
        proj = {"name": "p1", "groups": {"ctl": [s["A1"]["path"]], "drug": [s["B1"]["path"], s["A1"]["path"]]},
                "meta": {"note": "t"}}
        core.save_project(root, proj)
        assert core.load_project(root, "p1") == {**proj, "meta": core.clean_meta(proj)} and core.list_projects(root) == ["p1"]  # save/load round trip
        try:
            core.save_project(root, {"name": "../x", "groups": {}})
            raise SystemExit("bad project name accepted")
        except ValueError:
            pass
        f = core.frequency(root, proj)
        for g, r in f.items():
            assert r["ok"] and abs(sum(r["pct"].values()) - 100) < 1e-9, (g, r)  # every group sums to 100
        assert f["drug"]["n"] == 37
        assert all(r["same_fit"] for r in f.values())
        mk(root / "b", "B2", 4, [0, 1, 1, 2], fit_id="OTHER")  # labels from a different fit must not be compared
        mixed = core.frequency(root, {"groups": {"x": [s["A1"]["path"]], "y": ["b/B2_sleap_1423.json.gz"]}})
        assert not any(r["ok"] for r in mixed.values()), mixed
        assert [len(p) for p in core.split_by_lengths(list(range(9)), [3, 4, 2])] == [3, 4, 2]
        try:
            core.split_by_lengths(list(range(8)), [3, 4, 2])
            raise SystemExit("length mismatch not caught")
        except AssertionError:
            pass
        bad = core.frequency(root, {"groups": {"x": [s["C1"]["path"]]}})  # unlabeled session must flag, not pass
        assert not bad["x"]["ok"] and bad["x"]["missing"]
        for i in range(5):  # LRU cap holds
            core.load(root, s["A1"]["path"]), core.load(root, s["B1"]["path"]), core.load(root, s["C1"]["path"])
        assert len(core._CACHE) <= core.CACHE_MAX
    with tempfile.TemporaryDirectory() as d2:  # R6: metadata survives a save/load round trip; strangers and long text are cleaned
        r2 = Path(d2)
        core.save_project(r2, {"name": "p", "groups": {"g": ["a.json.gz"]}, "meta": {"description": "x" * 300, "group_notes": {"g": "n", "zz": "gone"},
                              "sessions": {"a.json.gz": {"animal": "m1", "condition": "cort", "date": "2023-12-05", "bad": 1}, "other": {"animal": "q"}}}})
        mm = core.load_project(r2, "p")["meta"]
        assert len(mm["description"]) == 200 and mm["group_notes"] == {"g": "n"} and list(mm["sessions"]) == ["a.json.gz"], mm
        assert mm["sessions"]["a.json.gz"] == {"animal": "m1", "condition": "cort", "date": "2023-12-05"}
    core.RESULTS_BASE = None  # results folder rules: default under the data root, overridable, commit hash recorded
    assert core.results_dir(Path("/d"), "p") == Path("/d/.subtle_results/p")
    core.RESULTS_BASE = Path("/o"); assert core.results_dir(Path("/d"), "p") == Path("/o/p"); core.RESULTS_BASE = None
    assert core.code_commit() != ""
    import clean
    with tempfile.TemporaryDirectory() as d3:  # retention: 5 runs, latest 3 kept, one older run protected by KEEP
        b = Path(d3)
        for n in ("260101_000000_a", "260102_000000_a", "260103_000000_a", "260104_000000_a", "260105_000000_a"):
            (b / "p" / n).mkdir(parents=True)
        (b / "p" / "260101_000000_a" / "KEEP").write_text("")
        assert [p.name for p in clean.stale(b)] == ["260102_000000_a"], clean.stale(b)
    print("selftest OK")


def stats_checks() -> None:
    """Hand-computed cases for stats.py and a known-rotation check for core.cam_view."""
    import numpy as np
    import stats
    a, b = np.array([1., 2, 3]), np.array([4., 5, 6])
    p, m = stats.perm_p(a, b)
    assert abs(p - 0.1) < 1e-9 and m == "exact permutation", (p, m)           # 2 of C(6,3)=20 splits are as extreme
    assert abs(stats.perm_p(np.array([1., 2, 3]), np.array([4., 5, 6, 7, 8]))[0] - 2 / 56) < 1e-9
    assert stats.hedges_g(a, b) < 0 and stats.hedges_g(a, a) == 0
    assert stats.bh([0.01, 0.04, 0.03]) == [0.03, 0.04, 0.04]
    assert [stats.stars(x) for x in (0.0004, 0.004, 0.04, 0.4)] == ["***", "**", "*", "ns"]
    rng = np.random.default_rng(0)
    th = 0.7
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])   # rows: right, up, toward viewer
    x3 = rng.normal(size=(800, 11, 3))
    cw, ch = core.CELLS[1][2:]
    u, v = x3 @ R[0], -(x3 @ R[1])                                               # image y points down
    kp = np.stack([(u * 100 + 600) / cw, (v * 100 + 600) / ch, np.ones_like(u)], -1)
    core.kp_cell = lambda s, cam: kp                                              # inject the synthetic 2D track
    res = core.cam_view({"xyz": x3}, 1)
    assert np.allclose(np.array(res["R"]), R, atol=1e-6), res["R"]
    print("stats + cam_view OK")


def dek_checks() -> None:
    """Welch t by hand, feature layout vs Kinematics output, and a planted speed effect that DEK must find (and must not invent)."""
    import numpy as np
    import dek
    t, p = dek.welch(np.array([[1.], [2.], [3.]]), np.array([[2.], [4.], [6.], [8.]]))
    assert abs(t[0] + 3 / 2 ** 0.5) < 1e-9, t                                   # (2 - 5) / sqrt(1/3 + (20/3)/4)
    nodes = ["nose1", "neck1", "earL1", "earR1", "forelegL1", "forelegR1", "tailstart1", "hindlegL1", "hindlegR1", "tail1", "tailend1"]
    edges = [[1, 4], [1, 5], [1, 0], [1, 3], [1, 2], [6, 8], [6, 7], [6, 9], [6, 1], [9, 10]]
    nd, ed, an, cols = dek.layout(nodes, edges)
    assert (len(nd), len(ed), len(an), len(cols)) == (11, 10, 7, 84) and cols[0] == "V0:nose" and cols[11] == "E0:neck-forelegL"
    rng = np.random.default_rng(1)

    def animal(speed: float) -> dict:
        x = np.cumsum(rng.normal(size=(600, 11, 3)) * 0.01 * speed, 0) + rng.normal(size=(1, 11, 3))  # random walk per node
        return {"xyz": x, "nodes": nodes, "edges": edges, "sub": None, "super": np.array([0, 1] * 300)}
    def table(speed): return dek.animal_table(animal(speed), "super", None)
    ids, M, n, c2 = table(1.0)
    assert list(ids) == [0, 1] and M.shape == (2, 84) and list(n) == [300, 300] and c2 == cols
    v1 = cols.index("V1:nose")
    fast = np.array([table(3.0)[1][0] for _ in range(4)]); slow = np.array([table(1.0)[1][0] for _ in range(4)])
    tt, pp = dek.welch(fast, slow)
    assert pp[v1] < 0.001 and tt[v1] > 0, (pp[v1], tt[v1])                       # planted effect found
    same_a = np.array([table(1.0)[1][0] for _ in range(4)]); same_b = np.array([table(1.0)[1][0] for _ in range(4)])
    assert dek.welch(same_a, same_b)[1][v1] > 0.01                               # no effect planted -> none claimed
    print("dek checks OK")


def stability_checks() -> None:
    """level_result on labels with a hand-known answer: 3 vs 3 animals perfectly separated -> exact p 2/20 for each cluster."""
    import numpy as np
    import stability
    lab = np.concatenate([np.array([0] * 8 + [1] * 2)] * 3 + [np.array([0] * 2 + [1] * 8)] * 3)
    r = stability.level_result(lab, [10] * 6, [[0, 1, 2], [3, 4, 5]], ["a", "b"])
    assert r["k"] == 2 and abs(r["min_p_adj"] - 0.1) < 1e-9 and r["n_sig"] == 0, r
    # runs cut at the session border: [0,0,1 | 1,1,2] -> runs 2,1 | 2,1 (the two 1s are not joined)
    w = stability.runs_of(np.array([0, 0, 1, 1, 1, 2]), [3, 3])
    assert w == {"median_run": 1.5, "mean_run": 1.5, "frac_runs_len1": 0.5}, w
    import model
    assert model.clustering_of(None) == "umap_phenograph" and model.clustering_of("pca_consensus") == "pca_consensus"
    print("stability checks OK")


def sweep_checks() -> None:
    """Hand-known p, a planted effect, and the null: no split of real-like noise may pass more often than 5 percent."""
    import numpy as np
    import sweep
    rng = np.random.default_rng(3)
    def labs(frac0):  # one animal: 300 frames, share frac0 in cluster 0 (+ small noise), rest cluster 1
        n0 = int(np.clip(frac0 + rng.normal(0, 0.01), 0, 1) * 300)
        return np.array([0] * n0 + [1] * (300 - n0))
    A = [labs(0.8) for _ in range(4)]; B = [labs(0.2) for _ in range(4)]
    r = sweep.sweep_levels({0: A + B}, [[0, 1, 2, 3], [4, 5, 6, 7]], ["a", "b"])
    assert r["exact"] and r["splits"] == 70 and abs(r["levels"][0]["p_level"] - 2 / 70) < 1e-9, r["levels"][0]["p_level"]   # observed split and its mirror
    assert r["recommended"]["significant"] == [0, 1], r["recommended"]                                                      # both clusters carry the planted difference
    N = [labs(0.5) for _ in range(8)]
    hits = 0
    for s in range(30):                                                         # null: same shares everywhere, random group assignment
        perm = np.random.default_rng(s).permutation(8)
        rr = sweep.sweep_levels({0: [N[i] for i in perm], 1: [N[i] for i in perm]}, [[0, 1, 2, 3], [4, 5, 6, 7]], ["a", "b"])
        hits += bool(rr["recommended"]["significant"])
    assert hits / 30 <= 0.15, hits
    print("sweep checks OK")


def calib_checks() -> None:
    """Synthetic 3-camera rig: distortion round trip, exact triangulation, and held-out reprojection error equal to the noise we add."""
    import numpy as np
    import calib
    rng = np.random.default_rng(5)
    cfg = {"camera_count": 3}
    for i, (rv, tv) in enumerate([([0.1, 0.0, 0.0], [0.0, 0.0, 1.0]), ([0.0, 0.4, 0.0], [-0.3, 0.0, 1.1]), ([0.0, -0.4, 0.1], [0.3, 0.0, 1.1])]):
        cfg[f"cam_{i + 1}"] = {"rotation": rv, "translation": tv, "matrix": [[1000, 0, 600], [0, 1000, 500], [0, 0, 1]], "distortions": [-0.1, 0.05, 0.001, -0.001, 0.0]}
    rig = calib.Rig(cfg)
    X = rng.normal(0, 0.05, (50, 3))
    px = rig.project(X, 0)
    assert np.abs(rig.undistort(px, 0) - (X @ rig.cams[0]["R"].T + rig.cams[0]["t"])[:, :2] / (X @ rig.cams[0]["R"].T + rig.cams[0]["t"])[:, 2:3]).max() < 1e-6   # distortion inverse
    P = [rig.undistort(rig.project(X, c), c) for c in range(3)]
    assert np.abs(np.array([rig.triangulate([P[c][n] for c in range(3)], [0, 1, 2]) for n in range(50)]) - X).max() < 1e-6   # exact triangulation
    print("calib checks OK")


if __name__ == "__main__":
    main()
    stats_checks()
    dek_checks()
    stability_checks()
    sweep_checks()
    calib_checks()
