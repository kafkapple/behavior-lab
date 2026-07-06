"""CalMS21 keypoint-count ablation: how many keypoints does behavior clustering need?

Follow-up to run_discovery_multiseed.py (must show stable ARI/silhouette across
seeds first, otherwise a count-vs-metric curve can't be told apart from seed noise).
Two independent removal schemes to avoid conflating "fewer keypoints" with
"which keypoints got removed":
  - anatomy_prior: nested removal, most dispensable first (ears -> hips -> neck),
    always keeping the nose-tailbase axis last.
  - random_paired: random keypoint subset per count (same subset for both mice),
    averaged over a few mask samples.

Inflection point (pre-registered): first per-animal keypoint count where ARI
drops >10% relative to the full (7-keypoint) mean, or by more than 2 std of the
Phase-A seed distribution -- whichever is looser.

Usage:
    python scripts/calms21_keypoint_ablation.py
"""
import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from run_discovery_multiseed import load_calms21, SEEDS  # also sets single-thread OMP env vars

import numpy as np
from behavior_lab.experiments.discovery import compare_discovery_methods
from behavior_lab.evaluation import compute_cluster_metrics

JOINT_NAMES = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]
ANATOMY_REMOVAL_ORDER = ["left_ear", "right_ear", "left_hip", "right_hip", "neck"]  # nose/tail_base kept last
N_MICE = 2
CV_SEEDS = SEEDS[:3]
MASK_SEEDS = [0, 1, 2]


def subset_indices(keep_names: list[str]) -> np.ndarray:
    """Per-animal keypoint name subset -> flat (N_MICE*len(keep_names),) index array."""
    keep_idx = [JOINT_NAMES.index(n) for n in keep_names]
    return np.array([m * len(JOINT_NAMES) + i for m in range(N_MICE) for i in keep_idx])


def anatomy_prior_subsets() -> dict[int, list[str]]:
    subsets = {}
    remaining = list(JOINT_NAMES)
    subsets[len(remaining)] = list(remaining)
    for name in ANATOMY_REMOVAL_ORDER:
        remaining.remove(name)
        subsets[len(remaining)] = list(remaining)
    return subsets  # per-animal counts 7,6,5,4,3,2


def run_config(kp: np.ndarray, gt: np.ndarray, keep_names: list[str], seed: int) -> dict:
    idx = subset_indices(keep_names)
    kp_sub = kp[:, idx, :]
    run = compare_discovery_methods(
        kp_sub, methods=["kmeans_pca_umap"], n_clusters=4, random_state=seed,
    )[0]
    m = compute_cluster_metrics(run.result.features, run.result.labels, gt)
    return {"ari": m.ari, "silhouette": m.silhouette}


def main():
    kp, gt = load_calms21()
    rng = np.random.default_rng(0)
    anatomy_subsets = anatomy_prior_subsets()

    results = []  # rows: scheme, n_keep_per_animal, seed, ari, silhouette

    for n_keep, keep_names in anatomy_subsets.items():
        for seed in CV_SEEDS:
            r = run_config(kp, gt, keep_names, seed)
            results.append({"scheme": "anatomy_prior", "n_keep": n_keep, **r})

    for n_keep in sorted(anatomy_subsets)[:-1]:  # skip the trivial full-set (=random has nothing to sample)
        if n_keep == len(JOINT_NAMES):
            continue
        for mask_seed in MASK_SEEDS:
            keep_names = list(rng.choice(JOINT_NAMES, size=n_keep, replace=False))
            r = run_config(kp, gt, keep_names, seed=mask_seed)
            results.append({"scheme": "random_paired", "n_keep": n_keep, **r})

    print(f"{'scheme':<14}{'n_keep/animal':>14}{'ARI mean':>10}{'ARI std':>9}"
          f"{'sil mean':>10}{'sil std':>9}  n_runs")
    summary = {}
    for scheme in ["anatomy_prior", "random_paired"]:
        for n_keep in sorted(set(r["n_keep"] for r in results if r["scheme"] == scheme), reverse=True):
            rows = [r for r in results if r["scheme"] == scheme and r["n_keep"] == n_keep]
            ari = np.array([r["ari"] for r in rows]); sil = np.array([r["silhouette"] for r in rows])
            summary[(scheme, n_keep)] = (ari.mean(), ari.std())
            print(f"{scheme:<14}{n_keep:>14}{ari.mean():>10.4f}{ari.std():>9.4f}"
                  f"{sil.mean():>10.4f}{sil.std():>9.4f}  {len(rows)}")

    full_ari_mean, full_ari_std = summary[("anatomy_prior", len(JOINT_NAMES))]
    print(f"\nFull-set (7 kp/animal) ARI = {full_ari_mean:.4f} +/- {full_ari_std:.4f} "
          f"(anatomy_prior baseline)")
    print("Inflection search (anatomy_prior; drop > max(10% relative, 2*full_std)):")
    threshold = max(abs(full_ari_mean) * 0.10, 2 * full_ari_std)
    inflection = None
    for n_keep in sorted({k[1] for k in summary if k[0] == "anatomy_prior"}, reverse=True):
        mean, _ = summary[("anatomy_prior", n_keep)]
        drop = full_ari_mean - mean
        flag = " <-- inflection" if drop > threshold and inflection is None else ""
        if flag:
            inflection = n_keep
        print(f"  n_keep={n_keep}: ARI={mean:.4f}  drop_from_full={drop:.4f}{flag}")
    if inflection is None:
        print("  No count crossed the pre-registered threshold -- flat or floor-effect-dominated curve.")


if __name__ == "__main__":
    main()
