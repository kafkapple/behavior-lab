"""SSL pretraining CLI with Hydra config.

Usage:
    python scripts/train_ssl.py ssl=dino model=infogcn dataset=mars training=default
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from behavior_lab.core import get_skeleton
from behavior_lab.models.ssl import build_ssl_model
from behavior_lab.data import SkeletonFeeder, get_feeder
from behavior_lab.training import Trainer, set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.experiment.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    skeleton = get_skeleton(cfg.dataset.skeleton)

    # Data
    data_dir = cfg.paths.data_dir
    train_set = get_feeder(
        cfg.dataset.name,
        data_path=f"{data_dir}/{cfg.dataset.name}_train.npz",
        split='train', window_size=cfg.dataset.window_size,
        debug=cfg.training.get('debug', False),
    )
    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # SSL Model
    ssl_cfg = cfg.get('ssl', {'method': 'dino'})
    model = build_ssl_model(
        encoder=cfg.model.name,
        ssl_method=ssl_cfg.get('method', 'dino'),
        in_channels=skeleton.num_channels,
        num_joints=skeleton.num_joints,
        num_frames=cfg.dataset.window_size,
        num_subjects=cfg.dataset.num_persons,
    )

    # Train
    train_cfg = {
        'optimizer': 'adam',
        'learning_rate': ssl_cfg.get('lr', 1e-3),
        'device': device,
        'output_dir': cfg.paths.output_dir,
        'seed': cfg.experiment.seed,
        **{k: v for k, v in cfg.training.items() if k not in ('name', 'optimizer')},
    }
    trainer = Trainer(model, train_loader, cfg=train_cfg)
    trainer.train()

    print(f"\nSSL pretraining complete. Model params: {model.num_parameters:,}")


if __name__ == "__main__":
    main()
