"""Unified trainer for supervised and SSL training."""
import time
import random
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Utilities ──


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Running average tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Stop training when metric stops improving."""

    def __init__(self, patience: int = 10, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None

    def __call__(self, score: float) -> bool:
        if self.best is None:
            self.best = score
            return False
        improved = score > self.best if self.mode == 'max' else score < self.best
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Checkpoint ──


def save_checkpoint(model, optimizer, epoch, metric, path, is_best=False):
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'epoch': epoch, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'metric': metric}
    torch.save(state, path)
    if is_best:
        torch.save(state, path.parent / 'best_model.pt')


def load_checkpoint(model, path, optimizer=None, device='cuda'):
    """Load model checkpoint, handling DataParallel prefix."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
    model.load_state_dict(sd, strict=False)
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return {'epoch': ckpt.get('epoch', 0), 'metric': ckpt.get('metric', 0.0)}


# ── LR Schedule ──


def build_lr_scheduler(optimizer, cfg):
    """Build LR scheduler from config."""
    steps = cfg.get('lr_steps', [70, 80])
    gamma = cfg.get('lr_decay_rate', 0.1)
    warmup = cfg.get('warmup_epochs', 5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=gamma)

    if warmup > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, scheduler], milestones=[warmup]
        )
    return scheduler


# ── Trainer ──


@dataclass
class TrainConfig:
    """Training configuration (maps to Hydra training/*.yaml)."""
    optimizer: str = 'sgd'
    learning_rate: float = 0.1
    weight_decay: float = 0.0004
    batch_size: int = 64
    num_epoch: int = 110
    warmup_epochs: int = 5
    lr_decay_rate: float = 0.1
    lr_steps: list = field(default_factory=lambda: [70, 80])
    nesterov: bool = True
    momentum: float = 0.9
    patience: int = 20
    seed: int = 42
    device: str = 'cuda'
    output_dir: str = 'outputs'
    resume: Optional[str] = None


class Trainer:
    """Unified trainer for graph models (supervised) and SSL models.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, cfg)
        trainer.train()
    """

    def __init__(self, model, train_loader, val_loader=None, cfg=None, loss_fn=None):
        cfg = cfg or TrainConfig()
        if isinstance(cfg, dict):
            tc = TrainConfig()
            for k, v in cfg.items():
                if hasattr(tc, k):
                    setattr(tc, k, v)
            cfg = tc

        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        self.is_ssl = hasattr(model, 'method')
        if self.is_ssl:
            self.loss_fn = None  # SSL models compute loss internally
        else:
            self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = build_lr_scheduler(self.optimizer, vars(cfg))
        self.early_stopping = EarlyStopping(patience=cfg.patience)

        # State
        self.start_epoch = 0
        self.best_metric = 0.0
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.resume:
            info = load_checkpoint(self.model, cfg.resume, self.optimizer, str(self.device))
            self.start_epoch = info['epoch']
            self.best_metric = info['metric']

    def _build_optimizer(self):
        cfg = self.cfg
        if cfg.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        return torch.optim.SGD(
            self.model.parameters(), lr=cfg.learning_rate,
            momentum=cfg.momentum, nesterov=cfg.nesterov, weight_decay=cfg.weight_decay
        )

    def train(self):
        """Run full training loop."""
        set_seed(self.cfg.seed)

        for epoch in range(self.start_epoch, self.cfg.num_epoch):
            train_loss = self._train_epoch(epoch)
            val_metric = self._validate(epoch) if self.val_loader else train_loss

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            is_best = val_metric > self.best_metric if not self.is_ssl else val_metric < self.best_metric
            if is_best:
                self.best_metric = val_metric

            save_checkpoint(
                self.model, self.optimizer, epoch, val_metric,
                self.output_dir / 'checkpoints' / f'epoch_{epoch:03d}.pt', is_best=is_best
            )

            print(f"[{epoch+1}/{self.cfg.num_epoch}] loss={train_loss:.4f} "
                  f"val={'acc' if not self.is_ssl else 'loss'}={val_metric:.4f} "
                  f"lr={lr:.6f} best={self.best_metric:.4f}")

            if self.early_stopping(val_metric if not self.is_ssl else -val_metric):
                print(f"Early stopping at epoch {epoch+1}")
                break

        return self.best_metric

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()

        for batch in tqdm(self.train_loader, desc=f'Train {epoch+1}', leave=False):
            if self.is_ssl:
                x = batch[0].float().to(self.device) if isinstance(batch, (list, tuple)) else batch.float().to(self.device)
                result = self.model(x)
                loss = result['loss'] if isinstance(result, dict) else result
            else:
                x, y = batch[0].float().to(self.device), batch[1].long().to(self.device)
                out = self.model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = self.loss_fn(logits, y)
                # InfoGCN MMD loss
                if isinstance(out, tuple) and len(out) >= 4:
                    from behavior_lab.models.losses import get_mmd_loss
                    loss = loss + get_mmd_loss(out[2], out[3])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_meter.update(loss.item(), x.size(0))

        return loss_meter.avg

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()

        if self.is_ssl:
            loss_meter = AverageMeter()
            for batch in self.val_loader:
                x = batch[0].float().to(self.device) if isinstance(batch, (list, tuple)) else batch.float().to(self.device)
                result = self.model(x)
                loss = result['loss'] if isinstance(result, dict) else result
                loss_meter.update(loss.item(), x.size(0))
            return loss_meter.avg

        correct = total = 0
        for batch in self.val_loader:
            x, y = batch[0].float().to(self.device), batch[1].long().to(self.device)
            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / total if total > 0 else 0.0
