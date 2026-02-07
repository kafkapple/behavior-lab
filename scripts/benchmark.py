"""Run benchmark: multiple discovery algorithms × datasets.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py benchmark.algorithms=[bsoid,moseq]
    python scripts/benchmark.py benchmark.datasets=[calms21,rat7m]
"""
import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    from behavior_lab.data.loaders import get_loader
    from behavior_lab.evaluation import (
        Evaluator, compute_behavior_metrics, compute_cluster_metrics,
    )

    output_dir = Path(cfg.paths.output_dir) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset_name in cfg.benchmark.datasets:
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {dataset_name}")
        log.info(f"{'='*60}")

        # Load dataset config
        dataset_cfg = OmegaConf.load(
            Path(cfg.config_dir) / "dataset" / f"{dataset_name}.yaml"
        ) if hasattr(cfg, 'config_dir') else cfg.get(dataset_name, {})

        try:
            loader = get_loader(
                dataset_name,
                data_dir=cfg.paths.data_dir + f"/{dataset_name}",
            )
            if hasattr(loader, 'load_split'):
                sequences = loader.load_split("train")
            else:
                sequences = loader.load_all()
                if isinstance(sequences, dict):
                    sequences = list(sequences.values())[0]
        except Exception as e:
            log.warning(f"Failed to load {dataset_name}: {e}")
            continue

        if not sequences:
            log.warning(f"No data for {dataset_name}")
            continue

        all_keypoints = np.concatenate([s.keypoints for s in sequences], axis=0)
        all_labels = None
        if sequences[0].labels is not None:
            all_labels = np.concatenate([s.labels for s in sequences], axis=0)

        fps = sequences[0].fps
        log.info(f"Frames: {all_keypoints.shape[0]}, Shape: {all_keypoints.shape}")

        for algo_name in cfg.benchmark.algorithms:
            log.info(f"\n--- Algorithm: {algo_name} ---")

            try:
                algo_cfg = OmegaConf.load(
                    Path("configs/model") / f"{algo_name}.yaml"
                )
                model = hydra.utils.instantiate(algo_cfg)
            except Exception as e:
                log.warning(f"Failed to instantiate {algo_name}: {e}")
                continue

            t0 = time.time()
            try:
                if hasattr(model, 'fit_predict'):
                    result = model.fit_predict(all_keypoints)
                    pred_labels = result.labels
                    features = result.features
                else:
                    fit_result = model.fit(all_keypoints)
                    if isinstance(fit_result, dict):
                        pred_labels = fit_result.get('labels', np.zeros(all_keypoints.shape[0]))
                        features = fit_result.get('features', None)
                    else:
                        pred_labels = model.predict(all_keypoints)
                        features = None
            except Exception as e:
                log.error(f"{algo_name} failed on {dataset_name}: {e}")
                continue

            elapsed = time.time() - t0
            n_clusters = len(set(pred_labels) - {-1})

            # Metrics
            behavior_m = compute_behavior_metrics(pred_labels, fps=fps)

            cluster_m = None
            if features is not None:
                try:
                    cluster_m = compute_cluster_metrics(features, pred_labels, all_labels)
                except Exception as e:
                    log.warning(f"Cluster metrics failed: {e}")

            result_entry = {
                "dataset": dataset_name,
                "algorithm": algo_name,
                "n_clusters": n_clusters,
                "n_frames": int(all_keypoints.shape[0]),
                "elapsed_sec": round(elapsed, 2),
                "temporal_consistency": round(behavior_m.temporal_consistency, 4),
                "entropy_rate": round(behavior_m.entropy_rate, 4),
                "num_bouts": behavior_m.num_bouts,
            }

            if cluster_m is not None:
                result_entry.update({
                    "nmi": round(cluster_m.nmi, 4),
                    "ari": round(cluster_m.ari, 4),
                    "silhouette": round(cluster_m.silhouette, 4),
                    "hungarian_accuracy": round(cluster_m.hungarian_accuracy, 4),
                    "davies_bouldin": round(cluster_m.davies_bouldin, 4),
                    "v_measure": round(cluster_m.v_measure, 4),
                })

            all_results.append(result_entry)
            log.info(f"  Clusters: {n_clusters}, Time: {elapsed:.1f}s")
            if cluster_m:
                log.info(
                    f"  NMI={cluster_m.nmi:.3f} ARI={cluster_m.ari:.3f} "
                    f"Silhouette={cluster_m.silhouette:.3f}"
                )

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nBenchmark results saved to {results_path}")

    # Print summary table
    log.info(f"\n{'Dataset':<12} {'Algorithm':<12} {'K':>4} {'NMI':>6} {'ARI':>6} {'Sil':>6} {'Time':>6}")
    log.info("-" * 60)
    for r in all_results:
        log.info(
            f"{r['dataset']:<12} {r['algorithm']:<12} {r['n_clusters']:>4} "
            f"{r.get('nmi', 0):>6.3f} {r.get('ari', 0):>6.3f} "
            f"{r.get('silhouette', 0):>6.3f} {r['elapsed_sec']:>5.1f}s"
        )


if __name__ == "__main__":
    main()
