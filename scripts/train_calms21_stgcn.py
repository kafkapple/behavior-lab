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


def stratified_val_split(subset, val_frac: float = 0.15, seed: int = 0):
    """Carve a class-balanced val split out of a training subset.

    Fixed seed (independent of the training seed) so the held-out val set -- and
    thus the model-selection target -- is identical across every run. Without this
    the trainer selected best_model.pt on the *test* set (val_loader==test_loader),
    leaking test into checkpoint selection.
    """
    base = subset.dataset if isinstance(subset, Subset) else subset
    base_idx = np.asarray(subset.indices) if isinstance(subset, Subset) else np.arange(len(subset))
    labels = np.asarray(base.label)[base_idx]
    rng = np.random.default_rng(seed)
    val_local = np.concatenate([
        rng.choice(np.flatnonzero(labels == c),
                   size=max(1, round((labels == c).sum() * val_frac)), replace=False)
        for c in np.unique(labels)
    ])
    val_mask = np.zeros(len(labels), dtype=bool); val_mask[val_local] = True
    return (Subset(base, base_idx[~val_mask].tolist()),
            Subset(base, base_idx[val_mask].tolist()))


def build_loaders(debug: bool, batch_size: int, max_train: int = None, max_test: int = None,
                   data_path: str = DEFAULT_DATA_PATH, val_frac: float = 0.15):
    train_set = get_feeder("calms21", data_path=data_path, split="train", debug=debug)
    test_set = get_feeder("calms21", data_path=data_path, split="test", debug=debug)
    train_subset = stratified_subset(train_set, max_train)
    test_subset = stratified_subset(test_set, max_test)
    train_final, val_subset = stratified_val_split(train_subset, val_frac)
    train_loader = DataLoader(train_final, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_set, test_set, train_loader, val_loader, test_loader


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
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda (Trainer falls back to cpu if cuda unavailable)")
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="null control: permute TRAIN labels only (val/test kept real) -> F1 should collapse to chance")
    args = ap.parse_args()

    max_train = None if args.debug else args.max_train
    max_test = None if args.debug else args.max_test
    train_set, test_set, train_loader, val_loader, test_loader = build_loaders(
        args.debug, args.batch_size, max_train, max_test, args.data_path)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} (held-out for "
          f"model selection) | Test: {len(test_loader.dataset)} seqs")

    if args.shuffle_labels:
        # Null control: shuffle ONLY the train-subset labels (val/test untouched, so model
        # selection + final eval use real labels). A classifier that learns real structure
        # from randomized labels can't -> F1 should collapse to ~chance. getitem reads
        # self.label[index] live, so mutating it here is enough.
        sub = train_loader.dataset  # Subset(train_set, train_idx)
        ti = np.asarray(sub.indices)
        sub.dataset.label[ti] = np.random.default_rng(args.seed).permutation(sub.dataset.label[ti])
        print(f"[null-control] shuffled {len(ti)} train labels (seed {args.seed})")

    # Trainer.train() seeds torch/numpy on its first line -- too late to control weight
    # init, since the model below is already constructed by then. Seed here instead.
    set_seed(args.seed)
    model = STGCN(num_classes=4, num_joints=7, num_persons=2, in_channels=2, skeleton="calms21")

    # val_loader (held out from train), NOT test_loader -- best_model.pt must be selected on
    # data the final test eval never sees, else checkpoint selection leaks the test set.
    trainer = Trainer(model, train_loader, val_loader, cfg={
        "device": args.device, "num_epoch": args.epochs, "batch_size": args.batch_size,
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
    eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    load_checkpoint(model, f"{args.output_dir}/checkpoints/best_model.pt", device=str(eval_device))
    model.to(eval_device)

    metrics, test_labels = evaluate(model, test_loader, eval_device)
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
