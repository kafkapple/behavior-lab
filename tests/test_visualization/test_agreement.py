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


def test_shift_stays_inside_each_recording():
    from behavior_lab.visualization.agreement import match_clusters, shift_within

    lab = np.r_[np.zeros(100, int), np.ones(100, int)]          # each recording = one cluster
    assert shift_within(lab, 0.3, [100, 100]).tolist() == lab.tolist()
    assert shift_within(lab, 0.3).tolist() != lab.tolist()      # a whole-sequence roll mixes them
    # two methods whose clusters are just the recording: not significant under the right null
    m = match_clusters(lab, lab + 5, n_shifts=50, lengths=[100, 100])
    assert m["p_mean"] > 0.5 and all(d["p"] > 0.5 for d in m["pairs"])


def test_meta_groups_and_recording_purity():
    from behavior_lab.visualization.agreement import match_clusters
    from behavior_lab.visualization.correspondence import (
        meta_groups,
        plot_meta_graph,
        plot_recording_shares,
        recording_purity,
    )

    rng = np.random.default_rng(0)
    a = np.repeat(rng.integers(0, 4, 300), 20)
    seqs = {"a": a, "b": (a + 1) % 4 + 10, "c": np.repeat(rng.integers(0, 3, 300), 20)}
    matches = {(x, y): match_clusters(seqs[x], seqs[y], n_shifts=50)
               for x, y in [("a", "b"), ("a", "c"), ("b", "c")]}
    share, groups = meta_groups(matches)
    ab = [g for g in groups if {n[0] for n in g} >= {"a", "b"}]
    assert 1 <= len(ab) <= 4                 # clusters of a are joined to their twins in b
    assert all(sum(n[0] == "a" for n in g) == sum(n[0] == "b" for n in g) for g in ab)
    assert abs(sum(v for n, v in share.items() if n[0] == "a") - 1) < 1e-9
    assert plot_meta_graph(matches, "demo").axes
    nmi, top = recording_purity(np.r_[np.zeros(50, int), np.ones(50, int)], [50, 50])
    assert nmi > 0.99 and top == 1.0         # clusters = recordings
    nmi, top = recording_purity(np.tile([0, 1], 50), [50, 50])
    assert nmi < 0.01 and top == 0.5         # clusters shared equally
    assert plot_recording_shares({"m": np.tile([0, 1], 50)}, [50, 50], ["r_1", "r_2"]).axes
