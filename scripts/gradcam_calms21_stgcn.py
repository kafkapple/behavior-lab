"""Grad-CAM keypoint-importance for the ST-GCN CalMS21 classifier.

Standard Grad-CAM (Selvaraju et al. 2017) adapted the way STG-Grad-CAM
(Das & Ortega, ICASSP 2022) adapts it for skeleton-GCNs: hook the last
graph-conv block's activation A (N*M, C, T, V), backprop the predicted-class
logit to get per-channel weights alpha_k = mean_{t,v}(dY/dA_k), then
CAM = ReLU(sum_k alpha_k * A_k) -> (N*M, T, V). Because the STGCN block output
is already indexed by (time, joint), the CAM *is* a per-keypoint, per-frame
importance map -- no extra aggregation trick needed for the "time-varying"
requirement.

Cross-checked against the model-agnostic occlusion-importance script from the
same session (calms21_keypoint_ablation.py) as a sanity signal, not a replacement:
occlusion answers "does removing this keypoint hurt the *clustering* metric",
Grad-CAM answers "which keypoint/frame did *this trained classifier* attend to
for *this predicted class*" -- different questions, both worth reporting.

Requires scripts/train_calms21_stgcn.py to have found real signal first
(F1-macro clearly above majority baseline) -- Grad-CAM on an undertrained/
chance-level classifier is a noise map, not an explanation.

Usage:
    python scripts/gradcam_calms21_stgcn.py
"""
import sys; sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from behavior_lab.core.skeleton import get_skeleton
from behavior_lab.data.feeders.skeleton_feeder import get_feeder
from behavior_lab.data.loaders.calms21 import CLASS_NAMES
from behavior_lab.models.graph.baselines import STGCN
from behavior_lab.training.trainer import load_checkpoint

DEFAULT_DATA_PATH = "data/calms21/calms21_aligned.npz"
DEFAULT_OUTPUT_DIR = "outputs/calms21_stgcn"
JOINT_NAMES = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]


class STGradCAM:
    """Grad-CAM on the last STGCNBlock -- output is already (N*M, C, T, V)."""

    def __init__(self, model: STGCN):
        self.model = model
        self.activations = self.gradients = None
        target = model.layers[-1]
        target.register_forward_hook(lambda m, i, o: setattr(self, "activations", o))
        target.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0]))

    def compute(self, x: torch.Tensor, target_class: torch.Tensor | None = None):
        """x: (N, C, T, V, M). Returns cam (N, M, T, V) in [0,1], pred (N,)."""
        self.model.zero_grad()
        logits = self.model(x)
        pred = logits.argmax(dim=1) if target_class is None else target_class
        logits[torch.arange(len(logits)), pred].sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # (N*M, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1))            # (N*M, T, V)
        N, M = x.shape[0], x.shape[-1]
        cam = cam.view(N, M, cam.shape[-2], cam.shape[-1])
        # Normalize per-(sample, mouse), not per-sample -- otherwise mouse0's scale gets
        # contaminated by mouse1's activation magnitude, biasing the cross-sample average
        # we report as "mouse0 keypoint importance" (code-review finding, 260706).
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), pred.detach().cpu().numpy()


def per_class_importance(model: STGCN, loader) -> dict:
    """Average Grad-CAM per predicted class -> {class_idx: (T, V) importance}."""
    gradcam = STGradCAM(model)
    sums, counts = {}, {}
    for x, _, _ in loader:
        cam, pred = gradcam.compute(x.float())
        cam_mouse0 = cam[:, 0]  # (N, T, V) -- report mouse-0's own joints for readability
        for i, c in enumerate(pred):
            sums[c] = sums.get(c, 0.0) + cam_mouse0[i]
            counts[c] = counts.get(c, 0) + 1
    return {c: sums[c] / counts[c] for c in sums}, counts


def plot_time_varying_importance(importance: np.ndarray, class_name: str, edges: list, out_path: str):
    """importance: (T, V). Color-coded skeleton strip across representative frames."""
    T, V = importance.shape
    frame_idx = np.linspace(0, T - 1, 6).astype(int)
    fig, axes = plt.subplots(1, len(frame_idx), figsize=(3 * len(frame_idx), 3.2))
    vmax = importance.max() + 1e-8
    xy = np.zeros((V, 2))  # keypoints have no fixed layout here -> use a canonical radial layout
    angles = np.linspace(0, 2 * np.pi, V, endpoint=False)
    xy[:, 0], xy[:, 1] = np.cos(angles), np.sin(angles)

    for ax, t in zip(axes, frame_idx):
        vals = importance[t] / vmax
        for a, b in edges:
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], "k-", lw=1, alpha=0.4, zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap="inferno", vmin=0, vmax=1,
                   s=250, edgecolors="black", linewidths=0.5, zorder=2)
        for k, name in enumerate(JOINT_NAMES):
            ax.annotate(name, xy[k], fontsize=6, ha="center", va="bottom")
        ax.set_title(f"t={t}", fontsize=9)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle(f"Grad-CAM keypoint importance over time -- predicted class: {class_name}", fontsize=11)
    fig.colorbar(cm.ScalarMappable(cmap="inferno"), ax=axes, shrink=0.6, label="normalized importance")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--ego_center", action="store_true",
                    help="MUST match the train run's --ego_center, else Grad-CAM sees a different input frame")
    args = ap.parse_args()
    ckpt_path = f"{args.output_dir}/checkpoints/best_model.pt"

    test_set = get_feeder("calms21", data_path=args.data_path, split="test", ego_center=args.ego_center)
    loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    model = STGCN(num_classes=4, num_joints=7, num_persons=2, in_channels=2, skeleton="calms21")
    load_checkpoint(model, ckpt_path, device="cpu")
    model.eval()

    # Grad-CAM needs gradients w.r.t. activations, not parameters -- keep params frozen either way
    for p in model.parameters():
        p.requires_grad_(True)

    importance, counts = per_class_importance(model, loader)
    edges = get_skeleton("calms21").edges

    print("Predicted-class counts in test set:", {CLASS_NAMES[c]: n for c, n in counts.items()})
    for c, imp in importance.items():
        out_path = f"{args.output_dir}/gradcam_{CLASS_NAMES[c]}.png"
        plot_time_varying_importance(imp, CLASS_NAMES[c], edges, out_path)
        ranked = sorted(zip(JOINT_NAMES, imp.mean(axis=0)), key=lambda kv: -kv[1])
        print(f"\n{CLASS_NAMES[c]} (n={counts[c]}) -- mean joint importance ranking:")
        for name, val in ranked:
            print(f"  {name}: {val:.3f}")


if __name__ == "__main__":
    main()
