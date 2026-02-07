"""Evaluation CLI: load checkpoint and evaluate on test set.

Usage:
    python scripts/evaluate.py dataset=mars model=infogcn \
        checkpoint=outputs/checkpoints/best_model.pt
"""
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.utils.data import DataLoader

from behavior_lab.core import get_skeleton, Graph
from behavior_lab.models import get_model
from behavior_lab.data import get_feeder
from behavior_lab.training import load_checkpoint
from behavior_lab.evaluation import Evaluator, compute_classification_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    skeleton = get_skeleton(cfg.dataset.skeleton)
    graph = Graph(skeleton)

    # Test data
    data_dir = cfg.paths.data_dir
    test_set = get_feeder(
        cfg.dataset.name,
        data_path=f"{data_dir}/{cfg.dataset.name}_test.npz",
        split='test', window_size=cfg.dataset.window_size,
    )
    test_loader = DataLoader(test_set, batch_size=cfg.training.batch_size, shuffle=False)

    # Model
    model = get_model(
        cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        num_joints=skeleton.num_joints,
        num_channels=skeleton.num_channels,
        num_persons=cfg.dataset.num_persons,
        A=torch.tensor(graph.A, dtype=torch.float32),
        **{k: v for k, v in cfg.model.items() if k not in ('name', '_target_')},
    )

    # Load checkpoint
    ckpt_path = cfg.get('checkpoint', 'outputs/checkpoints/best_model.pt')
    load_checkpoint(model, ckpt_path, device=device)
    model = model.to(device)
    model.eval()

    # Predict
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].float().to(device), batch[1]
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Report
    evaluator = Evaluator(class_names=cfg.dataset.get('class_names'))
    metrics = evaluator.evaluate_supervised(y_true, y_pred)
    evaluator.print_report(metrics)


if __name__ == "__main__":
    main()
