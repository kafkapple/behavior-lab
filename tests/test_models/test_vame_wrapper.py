"""Wiring + contract tests for the VAME discovery wrapper.

These do NOT require the heavy `vame-py` package: construction and registry
wiring work without it, and the fit path is asserted to raise a clean
ImportError when the dependency is absent (matching the moseq/subtle pattern).
"""
import numpy as np
import pytest

from behavior_lab.models import get_model, list_models
from behavior_lab.data.features.catalog import list_discovery_methods


def test_vame_registered():
    assert "vame" in list_models()["discovery"]


def test_vame_constructs_with_params():
    m = get_model("vame", project_dir="/tmp/vame_test", n_clusters=8, latent_dim=16)
    assert type(m).__name__ == "VAME"
    assert m.n_clusters == 8
    assert m.latent_dim == 16
    for method in ("fit", "predict", "fit_predict", "get_embeddings", "save", "load"):
        assert hasattr(m, method), f"VAME missing {method}"


def test_vame_in_catalog():
    assert any(s.name == "VAME" for s in list_discovery_methods())


def test_vame_fit_without_dep_raises_importerror():
    try:
        import vame  # noqa: F401
        pytest.skip("vame-py installed; ImportError path not exercised")
    except ImportError:
        pass
    m = get_model("vame", project_dir="/tmp/vame_test")
    kp = np.random.default_rng(0).random((40, 9, 2)).astype("float32")
    with pytest.raises(ImportError):
        m.fit(kp)


def test_vame_dlc_csv_shim(tmp_path):
    """The (T,K,D)->DLC CSV shim should not require vame and be well-formed."""
    m = get_model("vame", project_dir=str(tmp_path))
    kp = np.arange(3 * 2 * 2, dtype="float32").reshape(3, 2, 2)
    csv_path = tmp_path / "rec.csv"
    m._write_dlc_csv(kp, csv_path, ["kp0", "kp1"])
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("scorer")
    assert lines[1].startswith("bodyparts")
    assert lines[2].startswith("coords")
    # 3 header rows + 3 data rows
    assert len(lines) == 6
    # each data row: frame idx + K*3 values
    assert len(lines[3].split(",")) == 1 + 2 * 3
