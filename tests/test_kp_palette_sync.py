"""Web app keypoint palette must equal the BehaviorSplatter region colours (REGION_RGB), per avatar_11 keypoint."""
import json
import os
import sys
from pathlib import Path

import pytest

BS_SRC = Path(os.environ.get("BS_SRC", Path.home() / "dev" / "BehaviorSplatter" / "src"))   # BS_SRC = a checkout with origin/dev
PALETTE = Path(__file__).resolve().parents[1] / "tools" / "subtle_viewer" / "web" / "keypoint_palette.json"


def test_web_palette_matches_registry():
    if not BS_SRC.is_dir():
        pytest.skip("BehaviorSplatter checkout not found")
    sys.path.insert(0, str(BS_SRC))
    sk = pytest.importorskip("behaviorsplatter.temporal_deform.skeletons")
    assert Path(sk.__file__).resolve().is_relative_to(BS_SRC.resolve()), f"imported {sk.__file__}, not BS_SRC"
    if not hasattr(sk, "region_palette_hex"):
        pytest.skip("SKIPPED, NOT CHECKED: BS_SRC has no region_palette_hex (older checkout); set BS_SRC=<BehaviorSplatter worktree at origin/dev>/src")
    hexes, spec, web = sk.region_palette_hex(), sk.SKELETONS["avatar_11"], json.loads(PALETTE.read_text())
    region = {i: r for r, _, idx in spec.region_colors for i in idx}
    got = {n: web[n][1].lower() for n in spec.keypoint_names}
    want = {n: hexes[region[i]] for i, n in enumerate(spec.keypoint_names)}
    assert got == want
