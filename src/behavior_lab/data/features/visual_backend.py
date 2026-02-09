"""DINOv2 visual feature backend.

Extracts dense visual features from video frames using pretrained DINOv2 models.
Requires: torch, torchvision  (install via `pip install behavior-lab[visual]`)
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

PoolMode = Literal["cls", "mean", "concat"]


class DINOv2Backend:
    """Extract dense visual features from video frames using DINOv2.

    Why DINOv2:
    - Pretrained on LVD-142M, no fine-tuning needed
    - Dense spatial features via patch tokens — effective for posture distinction
    - Single-line load via torch.hub, no repo cloning required

    Models & dimensions:
        dinov2_vits14 → 384D
        dinov2_vitb14 → 768D
        dinov2_vitl14 → 1024D
        dinov2_vitg14 → 1536D

    Pooling modes:
        cls   — CLS token only (1 vector per frame, global)
        mean  — mean of patch tokens (spatial info preserved)
        concat — CLS + mean concatenated (2x dim)
    """

    name = "dinov2"

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str = "cpu",
        pool: PoolMode = "cls",
        batch_size: int = 32,
    ):
        try:
            import torch
            import torchvision.transforms as T
        except ImportError as e:
            raise ImportError(
                "DINOv2Backend requires torch and torchvision. "
                "Install with: pip install behavior-lab[visual]"
            ) from e

        self._torch = torch
        self.device = device
        self.pool = pool
        self.batch_size = batch_size

        logger.info("Loading DINOv2 model: %s", model_name)
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )
        self.model.eval().to(device)

        # Standard ImageNet normalization for DINOv2
        self._transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self._embed_dim = self.model.embed_dim
        logger.info("DINOv2 ready: dim=%d, pool=%s, device=%s",
                     self._embed_dim, pool, device)

    @property
    def dim(self) -> int:
        if self.pool == "concat":
            return self._embed_dim * 2
        return self._embed_dim

    def _preprocess(self, frames: np.ndarray) -> "torch.Tensor":
        """Convert (N, H, W, 3) uint8 frames to normalized tensor."""
        torch = self._torch
        # (N, H, W, 3) uint8 → (N, 3, H, W) float [0, 1]
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        x = self._transform(x)
        return x

    def _pool_features(
        self, cls_token: "torch.Tensor", patch_tokens: "torch.Tensor"
    ) -> "torch.Tensor":
        """Apply pooling strategy to DINOv2 output tokens."""
        if self.pool == "cls":
            return cls_token
        elif self.pool == "mean":
            return patch_tokens.mean(dim=1)
        else:  # concat
            torch = self._torch
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)

    def _extract_batch(self, batch: "torch.Tensor") -> np.ndarray:
        """Run inference on a single batch (no_grad context)."""
        torch = self._torch
        with torch.no_grad():
            batch = batch.to(self.device)
            output = self.model.forward_features(batch)
            cls_token = output["x_norm_clstoken"]    # (B, D)
            patch_tokens = output["x_norm_patchtokens"]  # (B, N_patches, D)
            pooled = self._pool_features(cls_token, patch_tokens)
            return pooled.cpu().numpy()

    def extract(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        """Extract DINOv2 features from video frames.

        Args:
            data: (N, H, W, 3) uint8 RGB frames

        Returns:
            (N, dim) feature matrix
        """
        if data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError(
                f"Expected (N, H, W, 3) uint8 frames, got shape {data.shape}"
            )

        torch = self._torch
        tensor = self._preprocess(data)
        n_frames = tensor.shape[0]
        features_list: list[np.ndarray] = []

        for start in range(0, n_frames, self.batch_size):
            batch = tensor[start : start + self.batch_size]
            features_list.append(self._extract_batch(batch))

        return np.concatenate(features_list, axis=0)
