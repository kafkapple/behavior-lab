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


def test_pose_embedding_sizes_and_method_maps():
    from behavior_lab.visualization.cluster_map import (
        cluster_sizes,
        plot_cluster_sizes,
        plot_method_maps,
        pose_embedding,
    )

    rng = np.random.default_rng(0)
    kp = rng.normal(size=(400, 5, 3)).cumsum(axis=0)
    emb = pose_embedding(kp)
    assert emb.shape == (400, 2) and 0 < pose_embedding.explained <= 1
    seqs = {"a": np.repeat([0, 1, 1, 2], 100), "b": np.repeat([3, -1], 100)}  # b is 2-frame bins
    ids, share, noise = cluster_sizes(seqs["b"])
    assert ids.tolist() == [3] and share.tolist() == [0.5] and noise == 0.5
    assert len(plot_method_maps(emb, seqs).axes) == 2
    assert "top 3 = 100%" in plot_cluster_sizes(seqs).axes[0].get_title()


def test_schema_and_player_data():
    from behavior_lab.visualization.keypoint_schema import plot_keypoint_schema
    from behavior_lab.visualization.player import player_block, player_data

    rng = np.random.default_rng(0)
    kp = rng.normal(size=(200, 4, 3))
    fig = plot_keypoint_schema(kp, ["a", "b", "c", "d"], [(0, 1), (1, 2)], "demo")
    assert len(fig.axes) == 2
    lab = np.repeat(np.arange(10), 20)
    d = player_data(kp, [(0, 1)], rng.normal(size=(200, 2)), {"m": lab, "binned": lab[::2]}, fps=20)
    assert d["n"] == 100 and len(d["kp"]) == 100 * 4 * 2 and len(d["emb"]) == 200
    # one index vector: both sequences show the label of frame 2 * i at step i
    assert d["methods"][0]["labels"] == d["methods"][1]["labels"] == lab[::2].tolist()
    assert 'id="p1"' in player_block("p1", "demo", d)
