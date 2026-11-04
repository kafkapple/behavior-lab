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
    if not hasattr(sk, "region_palette_hex"):
        pytest.skip("BehaviorSplatter without region_palette_hex (older checkout)")
    hexes, spec, web = sk.region_palette_hex(), sk.SKELETONS["avatar_11"], json.loads(PALETTE.read_text())
    region = {i: r for r, _, idx in spec.region_colors for i in idx}
    got = {n: web[n][1].lower() for n in spec.keypoint_names}
    want = {n: hexes[region[i]] for i, n in enumerate(spec.keypoint_names)}
    assert got == want
