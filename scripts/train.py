"""Supervised training CLI with Hydra config.

Usage:
    python scripts/train.py dataset=mars model=infogcn training=default
    python scripts/train.py dataset=mars model=infogcn training=fast_debug
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from behavior_lab.core import get_skeleton, Graph
from behavior_lab.data import SkeletonFeeder, get_feeder
from behavior_lab.models import get_model
from behavior_lab.training import Trainer, set_seed
from behavior_lab.evaluation import Evaluator


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.experiment.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Skeleton + Graph
    skeleton = get_skeleton(cfg.dataset.skeleton)
    graph = Graph(skeleton)

    # Data
    data_dir = cfg.paths.data_dir
    train_set = get_feeder(
        cfg.dataset.name,
        data_path=f"{data_dir}/{cfg.dataset.name}_train.npz",
        split='train', window_size=cfg.dataset.window_size,
        random_rot=True, debug=cfg.training.get('debug', False),
    )
    val_set = get_feeder(
        cfg.dataset.name,
        data_path=f"{data_dir}/{cfg.dataset.name}_val.npz",
        split='test', window_size=cfg.dataset.window_size,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

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

    # Train
    train_cfg = {
        'device': device,
        'output_dir': cfg.paths.output_dir,
        'seed': cfg.experiment.seed,
        **{k: v for k, v in cfg.training.items() if k != 'name'},
    }
    trainer = Trainer(model, train_loader, val_loader, cfg=train_cfg)
    best_acc = trainer.train()

    # Evaluate
    print(f"\nBest accuracy: {best_acc:.4f}")
    evaluator = Evaluator(class_names=cfg.dataset.get('class_names'))
    print(f"Classes: {cfg.dataset.get('class_names', 'N/A')}")


if __name__ == "__main__":
    main()
