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


if __name__ == "__main__":
    main()
    stats_checks()
