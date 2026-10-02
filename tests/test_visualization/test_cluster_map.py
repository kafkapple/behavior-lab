import matplotlib

matplotlib.use("Agg")
import numpy as np

from behavior_lab.visualization.cluster_map import (
    plot_cluster_map,
    plot_subtle_cluster_map,
    transition_matrix,
)


def test_transition_matrix_drops_self_and_noise():
    labels = np.array([0, 0, 1, 1, -1, 2, 2, 0, 1])
    ids, P = transition_matrix(labels)
    assert ids.tolist() == [0, 1, 2]
    assert np.all(np.diag(P) == 0)
    assert P[0, 1] == 1.0 and P[2, 0] == 1.0 and P[1].sum() == 0  # 1 -> noise is not counted


def test_cluster_maps_render():
    rng = np.random.default_rng(0)
    sub = np.repeat(rng.integers(0, 6, 40), 10)
    emb = rng.normal(size=(len(sub), 2)) + sub[:, None]
    sup = np.stack([np.zeros_like(sub), sub // 3, sub // 2], axis=1)
    ax = plot_cluster_map(emb, sub, "demo")
    assert "6 clusters" in ax.get_title()
    fig = plot_subtle_cluster_map(emb, sub, sup, "demo")
    assert "3 clusters" in fig.axes[1].get_title()  # finest supercluster level
