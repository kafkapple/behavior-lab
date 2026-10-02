"""Label agreement: a finer split must show up as asymmetric homogeneity, not as disagreement."""
import numpy as np

from behavior_lab.visualization.agreement import label_agreement, stretch_labels


def test_split_vs_unrelated():
    rng = np.random.default_rng(0)
    coarse = np.repeat(rng.integers(0, 4, 60), 20)
    fine = coarse * 2 + (np.arange(len(coarse)) // 10 % 2)  # every coarse label split in two
    unrelated = rng.integers(0, 4, len(coarse))
    agr = label_agreement({"coarse": coarse, "fine": fine, "unrelated": unrelated})
    assert agr["homogeneity"][0, 1] > 0.99 > agr["homogeneity"][1, 0]  # fine sits inside coarse
    assert agr["ari"][0, 1] > 0.4 and abs(agr["ari"][0, 2]) < 0.05
    assert abs(agr["null_ari"][0, 1]) < 0.1 < agr["ari"][0, 1]


def test_stretch_handles_binned_labels():
    binned = np.repeat(np.arange(10) % 3, 5)  # 50 bins, bouts of 5 bins
    out = stretch_labels(binned, 100)
    assert len(out) == 100 and out[0] == binned[0] and out[-1] == binned[-1]
    agr = label_agreement({"native": np.repeat(binned, 2), "binned": binned})
    assert (out == np.repeat(binned, 2)).all()  # integer ratio: exact
    assert agr["ari"][0, 1] == 1.0


def test_stretch_labels_whole_frame_bins():
    from behavior_lab.visualization.agreement import stretch_labels

    bins = np.arange(6004)  # B-SOiD: 2-frame bins from frame 0, short tail dropped
    out = stretch_labels(bins, 12010)
    assert out[0] == 0 and out[1] == 0 and out[2] == 1
    assert out[12007] == 6003 and out[-1] == 6003  # tail keeps the last bin, no drift


def test_match_clusters_recovers_relabelled_split():
    from behavior_lab.visualization.agreement import match_clusters

    rng = np.random.default_rng(0)
    a = np.repeat(rng.integers(0, 4, 300), 20)
    b = (a + 5) % 4 + 10                     # the same segmentation under other ids
    b[a == 0] = np.where(np.arange((a == 0).sum()) % 2, 20, 21)  # cluster 0 split in two
    b[:50] = -1                              # noise is ignored
    m = match_clusters(a, b, n_shifts=50)
    exact = [d for d in m["pairs"] if d["a"] != 0]
    assert len(exact) == 3 and all(d["jaccard"] > 0.95 and d["p"] < 0.05 for d in exact)
    assert all(d["jaccard"] > d["expected"] for d in m["pairs"])
    half = next(d for d in m["pairs"] if d["a"] == 0)
    assert 0.4 < half["jaccard"] < 0.6
    assert m["p_mean"] < 0.05

    unrelated = match_clusters(a, np.repeat(rng.integers(0, 4, 300), 20), n_shifts=50)
    assert unrelated["p_mean"] > 0.05
