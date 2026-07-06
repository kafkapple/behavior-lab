"""Supervised ST-GCN on CalMS21's own GT behavior labels (4 classes).

Motivation: today's unsupervised KMeans-PCA-UMAP clustering got ARI~=0 on CalMS21
(chance level) regardless of keypoint count. This checks a sharper question with
the same data: can a real spatio-temporal graph model decode the 4 GT behavior
classes from the same 7-keypoint/animal pose data at all (supervised), using the
GCN family already implemented in models/graph/ but never wired into training?
CalMS21 has known class imbalance ("other" dominates) so accuracy alone is
misleading -- macro-F1 vs. a majority-class baseline is the real signal check
(same rigor lesson as bmae_probe_cv3).

Usage:
    python scripts/train_calms21_stgcn.py --epochs 8 --debug   # quick sanity/timing pass
    python scripts/train_calms21_stgcn.py --epochs 8           # full run
"""
import argparse
import time

import sys; sys.path.insert(0, "src")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from behavior_lab.data.feeders.skeleton_feeder import get_feeder
from behavior_lab.data.loaders.calms21 import CLASS_NAMES
from behavior_lab.models.graph.baselines import STGCN
from behavior_lab.training import Trainer
from behavior_lab.training.trainer import set_seed, load_checkpoint
from behavior_lab.evaluation import compute_classification_metrics

DEFAULT_DATA_PATH = "data/calms21/calms21_aligned.npz"


def stratified_subset(dataset, max_n: int, seed: int = 0):
    """Class-balanced (not just first-N) subsample -- keeps rare classes represented."""
    if max_n is None or max_n >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.label)
    classes = np.unique(labels)
    per_class = max_n // len(classes)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=min(per_class, (labels == c).sum()), replace=False)
        for c in classes
    ])
    rng.shuffle(idx)
    return Subset(dataset, idx.tolist())


def build_loaders(debug: bool, batch_size: int, max_train: int = None, max_test: int = None,
                   data_path: str = DEFAULT_DATA_PATH):
    train_set = get_feeder("calms21", data_path=data_path, split="train", debug=debug)
    test_set = get_feeder("calms21", data_path=data_path, split="test", debug=debug)
    train_subset = stratified_subset(train_set, max_train)
    test_subset = stratified_subset(test_set, max_test)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_set, test_set, train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y, _ in loader:
        out = model(x.float().to(device))
        logits = out[0] if isinstance(out, tuple) else out
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    pred = logits.argmax(axis=1)
    return compute_classification_metrics(labels, pred, CLASS_NAMES), labels


def majority_baseline_f1(labels: np.ndarray) -> float:
    majority = np.bincount(labels).argmax()
    pred = np.full_like(labels, majority)
    return compute_classification_metrics(labels, pred, CLASS_NAMES).f1_macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_train", type=int, default=4000, help="class-balanced subsample (CPU wall-clock control)")
    ap.add_argument("--max_test", type=int, default=1200)
    ap.add_argument("--debug", action="store_true", help="100-sample subset, for timing/sanity only")
    ap.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument("--output_dir", type=str, default="outputs/calms21_stgcn")
    ap.add_argument("--seed", type=int, default=42, help="Trainer seed (weight init + shuffle order)")
    args = ap.parse_args()

    max_train = None if args.debug else args.max_train
    max_test = None if args.debug else args.max_test
    train_set, test_set, train_loader, test_loader = build_loaders(
        args.debug, args.batch_size, max_train, max_test, args.data_path)
    print(f"Train: {len(train_loader.dataset)}/{len(train_set)} seqs (subsampled) | "
          f"Test: {len(test_loader.dataset)}/{len(test_set)} seqs")

    # Trainer.train() seeds torch/numpy on its first line -- too late to control weight
    # init, since the model below is already constructed by then. Seed here instead.
    set_seed(args.seed)
    model = STGCN(num_classes=4, num_joints=7, num_persons=2, in_channels=2, skeleton="calms21")

    trainer = Trainer(model, train_loader, test_loader, cfg={
        "device": "cpu", "num_epoch": args.epochs, "batch_size": args.batch_size,
        "output_dir": args.output_dir, "patience": args.epochs,  # no early stop mid-probe
        "learning_rate": 0.1, "warmup_epochs": 1, "momentum": 0.9, "nesterov": True,
        "weight_decay": 0.0004, "lr_decay_rate": 0.1, "seed": args.seed,
        "lr_steps": [max(2, args.epochs - 4), max(3, args.epochs - 2)],  # fast_debug-style late decay
    })

    t0 = time.time()
    best_acc = trainer.train()
    elapsed = time.time() - t0
    print(f"\nTrained {args.epochs} epochs in {elapsed:.1f}s ({elapsed/args.epochs:.1f}s/epoch). "
          f"Trainer best val acc={best_acc:.4f}")

    # trainer.train() only checkpoints the best epoch to disk -- self.model is left at
    # the LAST epoch's weights, which can be worse (no early stopping here). Reload best
    # before reporting/evaluating, so this script's own numbers match what a later
    # separate gradcam_calms21_stgcn.py run (which always loads best_model.pt) sees.
    load_checkpoint(model, f"{args.output_dir}/checkpoints/best_model.pt", device="cpu")

    metrics, test_labels = evaluate(model, test_loader, torch.device("cpu"))
    baseline_f1 = majority_baseline_f1(test_labels)
    print(f"\nTest accuracy: {metrics.accuracy:.4f}")
    print(f"Test F1-macro: {metrics.f1_macro:.4f}  (majority-class baseline: {baseline_f1:.4f})")
    for name, f1 in metrics.f1_per_class.items():
        print(f"  {name}: F1={f1:.4f}")
    signal = metrics.f1_macro - baseline_f1
    print(f"\nSignal over majority baseline: {signal:+.4f} "
          f"({'REAL SIGNAL' if signal > 0.05 else 'NO MEANINGFUL SIGNAL -- Grad-CAM would be noise'})")


if __name__ == "__main__":
    main()
