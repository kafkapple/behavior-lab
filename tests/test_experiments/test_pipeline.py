"""Tests for the unified cross-species comparison pipeline."""
import numpy as np

from behavior_lab.experiments import DATASET_SPECS, run_comparison


def test_run_comparison_array_with_gt_and_isolated(tmp_path):
    rng = np.random.default_rng(0)
    kp = rng.random((300, 14, 2)).astype("float32")
    gt = rng.integers(0, 4, 300)
    out = tmp_path / "r.html"
    results = run_comparison(
        kp, "calms21", out, methods=("kmeans_pca_umap",),
        extra_labels={"my_iso": np.repeat(rng.integers(0, 5, 30), 10)},
        ground_truth=gt)
    assert out.exists()
    assert "kmeans_pca_umap" in results and "my_iso" in results
    html = out.read_text()
    assert "ARI (GT)" in html and "CalMS21" in html


def test_dataset_specs_multispecies():
    species = {s.species for s in DATASET_SPECS.values()}
    assert {"mouse", "rat", "human"} <= species  # already generalizes across species
    assert DATASET_SPECS["nwucla"].species == "human"
    assert DATASET_SPECS["nwucla"].skeleton == "nwucla"
