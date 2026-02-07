"""Run a single behavior discovery algorithm on a dataset.

Usage:
    python scripts/discover.py model=bsoid dataset=calms21
    python scripts/discover.py model=moseq dataset=rat7m
"""
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from behavior_lab.data.loaders import get_loader
    from behavior_lab.evaluation import Evaluator, compute_behavior_metrics

    # Load dataset
    log.info(f"Loading dataset: {cfg.dataset.name}")
    loader = hydra.utils.instantiate(cfg.dataset)

    if hasattr(loader, 'load_split'):
        sequences = loader.load_split("train")
    elif hasattr(loader, 'load_all'):
        result = loader.load_all()
        sequences = result.get("train", list(result.values())[0] if result else [])
    else:
        raise ValueError(f"Loader {type(loader)} has no load method")

    if not sequences:
        log.error("No sequences loaded")
        return

    log.info(f"Loaded {len(sequences)} sequences")

    # Concatenate keypoints
    all_keypoints = np.concatenate([s.keypoints for s in sequences], axis=0)
    all_labels = None
    if sequences[0].labels is not None:
        all_labels = np.concatenate([s.labels for s in sequences], axis=0)

    log.info(f"Total frames: {all_keypoints.shape[0]}, shape: {all_keypoints.shape}")

    # Instantiate model
    log.info(f"Instantiating model: {cfg.model.name}")
    model = hydra.utils.instantiate(cfg.model)

    # Run discovery
    log.info("Running behavior discovery...")
    if hasattr(model, 'fit_predict'):
        result = model.fit_predict(all_keypoints)
        pred_labels = result.labels
    elif hasattr(model, 'fit'):
        fit_result = model.fit(all_keypoints)
        if isinstance(fit_result, dict):
            pred_labels = fit_result.get('labels', np.zeros(all_keypoints.shape[0]))
        else:
            pred_labels = model.predict(all_keypoints)
    else:
        raise ValueError(f"Model {type(model)} has no fit method")

    n_clusters = len(set(pred_labels) - {-1})
    log.info(f"Discovered {n_clusters} clusters")

    # Evaluate
    evaluator = Evaluator()

    # Behavior metrics
    fps = sequences[0].fps if sequences else 30.0
    behavior_metrics = compute_behavior_metrics(pred_labels, fps=fps)
    log.info(f"Temporal consistency: {behavior_metrics.temporal_consistency:.4f}")
    log.info(f"Num bouts: {behavior_metrics.num_bouts}")
    log.info(f"Entropy rate: {behavior_metrics.entropy_rate:.4f}")

    # Cluster metrics (if embeddings available)
    if hasattr(result, 'features') and result.features is not None:
        cluster_metrics = evaluator.evaluate_clusters(
            result.features, pred_labels, all_labels
        )
        report = evaluator.print_report(cluster_metrics)
        log.info(f"\n{report}")

    # Save results
    output_dir = Path(cfg.paths.output_dir) / "discovery" / cfg.model.name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "labels.npy", pred_labels)
    if hasattr(result, 'embeddings') and result.embeddings is not None:
        np.save(output_dir / "embeddings.npy", result.embeddings)

    log.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
