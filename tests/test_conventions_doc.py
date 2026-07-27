"""Drift guard: docs/conventions.md must agree with the code it points at.

The doc carries a ```yaml checked block of the facts it asserts. This test reads
each fact from the actual source and compares. Change the code without changing
the doc (or vice versa) and this fails — which is the whole point, since a
conventions doc that can silently go stale is worse than none.

Sibling repos (BehaviorSplatter, sdannce-poc) live outside this repo, so those
checks skip when the sibling is absent rather than failing the suite.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "conventions.md"
DEV = Path(__file__).resolve().parents[2]          # ~/dev
BS = DEV / "BehaviorSplatter"
SDP = DEV / "sdannce-poc"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checked() -> dict:
    m = re.search(r"```yaml checked\n(.*?)```", DOC.read_text(), re.S)
    assert m, "docs/conventions.md: no ```yaml checked block"
    out = {}
    for line in m.group(1).strip().splitlines():
        key, _, val = line.partition(":")
        out[key.strip()] = ast.literal_eval(val.strip())
    return out


def _list_len(path: Path, name: str) -> int:
    """Length of a module-level list literal, read without importing."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        # plain `NAME = [...]` and annotated `NAME: list[str] = [...]`
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(getattr(t, "id", None) == name for t in targets):
            return len(node.value.elts)
    raise AssertionError(f"{path.name}: {name} not found as a list literal")


def _skip_unless(path: Path):
    if not path.exists():
        pytest.skip(f"sibling repo missing: {path}")


def test_kp22(checked):
    _skip_unless(BS)
    src = BS / "src/behaviorsplatter/temporal_deform/keypoints_22.py"
    assert _list_len(src, "KP22_NAMES") == checked["kp22_names"]
    assert _list_len(src, "SKELETON_BONES") == checked["kp22_bones"]


def test_rat23(checked):
    _skip_unless(SDP)
    src = SDP / "src/sdannce_utils/constants.py"
    assert _list_len(src, "KP_NAMES") == checked["rat23_names"]
    assert _list_len(src, "SKELETON_EDGES") == checked["rat23_edges"]


def test_sbea16(checked):
    assert _list_len(REPO / "scripts/sbea_dlc_triangulate.py",
                     "BODYPARTS") == checked["sbea16_bodyparts"]
    assert _list_len(REPO / "scripts/render_sbea_report.py",
                     "EDGES") == checked["sbea16_edges"]


def test_m5_constants(checked):
    _skip_unless(BS)
    text = (BS / "src/behaviorsplatter/notebooks/kp_utils.py").read_text()
    centre = re.search(r"M5_SCENE_CENTER\s*=\s*np\.array\(\[([^\]]+)\]", text)
    assert centre, "kp_utils.py: M5_SCENE_CENTER literal not found"
    assert [float(x) for x in centre.group(1).split(",")] == checked["m5_scene_center"]

    scale = re.search(r"M5_DISTANCE_SCALE\s*=\s*([\d.]+)\s*/\s*([\d.]+)", text)
    assert scale, "kp_utils.py: M5_DISTANCE_SCALE expression not found"
    got = float(scale.group(1)) / float(scale.group(2))
    assert round(got, 9) == checked["m5_distance_scale"]


def test_tensor_format_api(checked):
    from behavior_lab.core import tensor_format
    for fn in checked["tensor_format_api"]:
        assert callable(getattr(tensor_format, fn, None)), f"missing {fn}"


def test_sbea_camera_order(checked):
    """Producer (this repo) and consumer (sdannce-poc config) must agree."""
    _skip_unless(SDP)
    text = (SDP / "configs/segmentation/sbea.yaml").read_text()
    m = re.search(r"camera_order:\s*\[([^\]]+)\]", text)
    assert m, "sbea.yaml: camera_order not set"
    assert [int(x) for x in m.group(1).split(",")] == checked["sbea_camera_order"]


def test_doc_paths_exist():
    """Every ~/dev-relative path the doc cites must resolve."""
    text = DOC.read_text()
    cited = {p for p in re.findall(r"`((?:behavior-lab|BehaviorSplatter|sdannce-poc|FaceLift)/[^`\s]+)`", text)
             if "::" not in p and not p.endswith((",", "."))}
    missing = sorted(p for p in cited if not (DEV / p).exists())
    assert not missing, f"docs/conventions.md cites missing paths: {missing}"
