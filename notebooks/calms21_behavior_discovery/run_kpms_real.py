"""Run the *real* keypoint-MoSeq ARHMM on CalMS21 dyadic data.

Replaces the GaussianHMM fallback used in compare_methods_v2.py with the
canonical sticky HDP-AR-HMM from kp-MoSeq 0.6.6 (jax-moseq backend).

Pipeline:
  1. Load CalMS21 raw JSON, slice resident mouse keypoints (T, 7, 2) per session
  2. kpms.format_data → pads + masks
  3. kpms.fit_pca → 8 PCs of aligned pose
  4. kpms.init_model + kpms.fit_model (AR(1) + sticky HDP, ~50 Gibbs sweeps)
  5. kpms.apply_model → per-frame syllable labels
  6. Score vs CalMS21 GT (ARI / NMI / weighted purity)
  7. Save labels + metric row alongside the v2 fallback for direct comparison

Output:
    data/calms21_behavior_discovery/results_v2/<latest>/kpms_real_labels.parquet
    data/calms21_behavior_discovery/results_v2/<latest>/kpms_real_metrics.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(k, "1")
os.environ["JAX_ENABLE_X64"] = "True"

import jax
jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ROOT = REPO / "data" / "calms21_behavior_discovery"
RAW_JSON = REPO / "data" / "calms21" / "task1_classic_classification" / "calms21_task1_train.json"
ANN = ROOT / "annotations"

BODYPARTS = ["nose", "left_ear", "right_ear", "neck",
             "left_hip", "right_hip", "tail_base"]
USE_BODYPARTS = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip"]
SKELETON_EDGES = [["nose", "neck"], ["left_ear", "neck"], ["right_ear", "neck"],
                  ["neck", "left_hip"], ["neck", "right_hip"]]


def load_resident_kp() -> dict[str, np.ndarray]:
    """Per-session resident keypoints (T, 7, 2)."""
    with open(RAW_JSON) as f:
        bundle = json.load(f)
    block = bundle[next(iter(bundle))]

    sessions = list(block.items())[:8]
    out = {}
    for key, seq in sessions:
        name = key.split("/")[-1]
        kp = np.array(seq["keypoints"], dtype=np.float32)   # (T, 2_mice, 2_xy, 7_kp)
        kp = kp.transpose(0, 1, 3, 2)                       # (T, 2_mice, 7_kp, 2_xy)
        out[name] = kp[:, 0, :, :]                          # resident
    return out


def main():
    import keypoint_moseq as kpms

    runs = sorted([p for p in (REPO / "outputs" / "calms21_behavior_discovery" / "results_v2").iterdir() if p.is_dir()])
    assert runs, "run compare_methods_v2.py first"
    run = runs[-1]
    print(f"v2 run: {run.name}")

    print(">>> loading resident keypoints from CalMS21 raw JSON")
    kp_per = load_resident_kp()
    for name, k in kp_per.items():
        print(f"  {name}: {k.shape}")

    print("\n>>> kpms.format_data")
    coords = kp_per                                          # dict session -> (T, K, 2)
    confs = {k: np.ones((v.shape[0], v.shape[1])) for k, v in coords.items()}
    data, metadata = kpms.format_data(
        coords, confs, bodyparts=BODYPARTS, use_bodyparts=USE_BODYPARTS,
        added_noise_level=0.1)
    print(f"  Y shape: {data['Y'].shape}  mask: {data['mask'].shape}")

    print("\n>>> kpms.fit_pca (8 components)")
    t0 = time.time()
    pca = kpms.fit_pca(
        Y=data["Y"], mask=data["mask"], conf=data["conf"],
        anterior_idxs=np.array([0, 1, 2, 3]),   # nose, ears, neck
        posterior_idxs=np.array([4, 5]),         # hips
    )
    print(f"  fit_pca done ({time.time()-t0:.1f}s)")

    print("\n>>> convert data + PCA components to float64 (jax x64 mode)")
    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data, x64=True)
    print(f"  Y dtype: {data['Y'].dtype}  conf dtype: {data['conf'].dtype}")

    # PCA components are float32 by default — cast to float64
    if hasattr(pca, "components_"):
        pca.components_ = pca.components_.astype(np.float64)
        pca.mean_ = pca.mean_.astype(np.float64)
        if hasattr(pca, "explained_variance_"):
            pca.explained_variance_ = pca.explained_variance_.astype(np.float64)
    print(f"  pca.components_ dtype: {pca.components_.dtype}")

    print("\n>>> kpms.init_model")
    import jax.random as jr
    model = kpms.init_model(
        data=data, pca=pca,
        anterior_idxs=np.array([0, 1, 2, 3]),
        posterior_idxs=np.array([4, 5]),
        whiten=True,
        seed=jr.PRNGKey(42),
        trans_hypparams={"num_states": 25, "kappa": 1e6,
                         "alpha": 5.7, "gamma": 1e3},
        ar_hypparams={"latent_dim": 8, "nlags": 3,
                      "S_0_scale": 0.01, "K_0_scale": 10.0},
        obs_hypparams={"nu_s": 5, "nu_sigma": 1e5,
                       "sigmasq_0": 0.1, "sigmasq_C": 0.1},
        cen_hypparams={"sigmasq_loc": 0.5},
        error_estimator={"intercept": 0.25, "slope": -0.5},
    )
    print(f"  states: {list(model['states'].keys())}  "
          f"params: {list(model['params'].keys())}")

    print("\n>>> kpms.fit_model (AR-HMM, 50 Gibbs sweeps)")
    t0 = time.time()
    model, model_name = kpms.fit_model(
        model, data, metadata, project_dir=str(run),
        model_name="kpms_real", num_iters=50, ar_only=False,
        generate_progress_plots=False,
        save_every_n_iters=50,
    )
    print(f"  fit_model done ({time.time()-t0:.1f}s)")

    print("\n>>> extract syllable labels from trained model states")
    # model['states']['z'] shape: (n_segments, max_T-nlags) — syllable id per frame
    # metadata = (recording_names, segments[(start, end)]) describes the slice
    # of each original session that each row of z covers.
    z_padded = np.asarray(model["states"]["z"])
    mask = np.asarray(data["mask"])
    print(f"  z padded: {z_padded.shape}  mask: {mask.shape}")

    recording_names = np.asarray(metadata[0])
    segs = np.asarray(metadata[1])   # (n_segments, 2) — [start, end] in original session
    nlags = 3   # matches ar_hypparams above

    # Aggregate per original session (each session may be split into multiple segments)
    sess_to_pieces: dict[str, list[tuple[int, np.ndarray]]] = {}
    for row_idx, (rec, (start, end)) in enumerate(zip(recording_names, segs)):
        # CalMS21 segments are contiguous (kpms pads only the tail) — assert this
        # invariant so a sparse mask from future data formats fails loudly.
        row_mask = mask[row_idx, nlags:]
        z_valid_len = int(row_mask.sum())
        assert row_mask[:z_valid_len].all(), (
            f"non-contiguous mask in {rec} — silent label truncation risk")
        z_row = z_padded[row_idx, :z_valid_len].astype(np.int32)
        sess_to_pieces.setdefault(str(rec), []).append((int(start), z_row))
        print(f"  {rec} [{start}:{end}]: {len(z_row)} z-frames  uniq={len(np.unique(z_row))}")

    sess_labels: dict[str, np.ndarray] = {}
    for sess, pieces in sess_to_pieces.items():
        pieces.sort(key=lambda p: p[0])
        # Pre-pad nlags with first label so total length matches original session
        head = np.full(nlags, pieces[0][1][0] if len(pieces[0][1]) else 0, dtype=np.int32)
        sess_labels[sess] = np.concatenate([head] + [p[1] for p in pieces])
    labels_per = sess_labels

    # Concatenate in the same order as labels_v2.parquet
    info = pd.read_csv(ROOT / "data_info_v2.csv")
    order = info["session"].tolist()
    kpms_labels = np.concatenate([labels_per[n] for n in order])
    gt = np.concatenate([np.load(ANN / f"{n}.npy")[:len(labels_per[n])]
                         for n in order]).astype(int)
    n = min(len(kpms_labels), len(gt))
    kpms_labels = kpms_labels[:n].astype(np.int32)
    gt = gt[:n]

    # Metrics
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score)
    ari = float(adjusted_rand_score(gt, kpms_labels))
    nmi = float(normalized_mutual_info_score(gt, kpms_labels))
    K = int(len(np.unique(kpms_labels)))
    print(f"\nkp-MoSeq REAL: K={K}  ARI={ari:.3f}  NMI={nmi:.3f}")

    pd.DataFrame([{"method": "kp-MoSeq_real", "k": K, "ari": ari, "nmi": nmi}]
                ).to_csv(run / "kpms_real_metrics.csv", index=False)

    # Merge with v2 labels parquet
    v2_labels = pd.read_parquet(run / "labels_v2.parquet")
    v2_labels["kpms_real"] = -1
    nn = min(len(v2_labels), len(kpms_labels))
    v2_labels.loc[:nn-1, "kpms_real"] = kpms_labels[:nn]
    v2_labels.to_parquet(run / "labels_v3.parquet")
    print(f"saved: {run / 'labels_v3.parquet'}")
    print(f"saved: {run / 'kpms_real_metrics.csv'}")


if __name__ == "__main__":
    main()
