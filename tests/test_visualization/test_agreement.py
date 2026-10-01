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
