"""Test the unified comparison-report module."""
import numpy as np

from behavior_lab.visualization.comparison import render_comparison_report


def test_render_comparison_report_dict_input(tmp_path):
    rng = np.random.default_rng(0)
    runs = {
        "kmeans": {"labels": rng.integers(0, 8, 300), "features": rng.random((300, 4))},
        "moseq": {"labels": np.repeat(rng.integers(0, 5, 30), 10)},
        "subtle": {"labels": np.repeat(rng.integers(0, 3, 60), 5), "metrics": {"silhouette": 0.12}},
    }
    out = render_comparison_report(runs, tmp_path / "cmp.html", fps=30.0, title="T")
    assert out.exists()
    html = out.read_text()
    for name in ("kmeans", "moseq", "subtle"):
        assert name in html
    assert "Metrics" in html
    assert "data:image/" in html  # at least the ethogram embedded
    # provided silhouette surfaces in the table
    assert "0.12" in html


def test_render_comparison_report_ground_truth(tmp_path):
    rng = np.random.default_rng(1)
    gt = rng.integers(0, 4, 200)
    runs = {
        "perfect": {"labels": gt.copy()},           # matches GT -> ARI ~1
        "random": {"labels": rng.integers(0, 4, 200)},
    }
    out = render_comparison_report(runs, tmp_path / "gt.html", fps=30.0, ground_truth=gt)
    html = out.read_text()
    assert "ARI (GT)" in html and "NMI (GT)" in html
    # perfect method should score ARI ~1.0
    assert "1.0" in html or "0.99" in html
