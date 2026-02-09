"""BehaveMAE wrapper: Hierarchical MAE for behavior analysis.

Reference: Stoffl et al. (ECCV 2024), "Elucidating the Hierarchical Nature of Behavior
           with Masked Autoencoders."

Setup (one of):
    1. Git submodule: git submodule add https://github.com/amathislab/BehaveMAE external/BehaveMAE
    2. Clone + PYTHONPATH: git clone ... && export PYTHONPATH=$PYTHONPATH:BehaveMAE
"""
import sys
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

# Auto-add submodule path if it exists
_SUBMODULE_PATH = Path(__file__).resolve().parents[4] / "external" / "BehaveMAE"
if _SUBMODULE_PATH.exists() and str(_SUBMODULE_PATH) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_PATH))


def pose_to_behavemae_input(data: np.ndarray, target_frames: int = 400) -> 'torch.Tensor':
    """Convert (T, K, D) skeleton data to BehaveMAE input format (B, T, 1, K*D).

    The model's forward() adds a channel dim via unsqueeze(1), so input should be 4D.

    Args:
        data: (T, K, D) skeleton coordinates
        target_frames: target temporal length (pad/crop)

    Returns:
        torch.Tensor of shape (1, T, 1, K*D)
    """
    import torch

    T, K, D = data.shape
    flat = data.reshape(T, K * D)  # (T, K*D)

    # Pad or crop to target_frames
    if T < target_frames:
        flat = np.pad(flat, ((0, target_frames - T), (0, 0)), mode='edge')
    elif T > target_frames:
        flat = flat[:target_frames]

    # (B, T, 1, K*D) — model.forward() adds channel dim via unsqueeze(1)
    tensor = torch.from_numpy(flat).float().unsqueeze(0).unsqueeze(2)
    return tensor


class BehaveMAE:
    """Wrapper for BehaveMAE hierarchical masked autoencoder.

    BehaveMAE learns multi-scale representations: lower levels capture fine-grained
    movemes, higher levels capture complex actions/activities.

    Requires the BehaveMAE repo to be cloned and importable.

    Usage:
        model = BehaveMAE.from_pretrained('path/to/checkpoint.pth', dataset='mabe22')
        features = model.encode(data)          # (T, K, D) -> multi-scale features
        features = model.encode_hierarchical(data)  # dict per level
    """

    # Default configs per dataset
    CONFIGS = {
        'shot7m2': dict(
            input_size=(400, 1, 72), in_chans=1, init_embed_dim=96,
            init_num_heads=2, stages=(2, 3, 4), out_embed_dims=(78, 128, 256),
            q_strides=[(2, 1, 4), (2, 1, 6)], mask_unit_attn=(True, False, False),
            patch_kernel=(2, 1, 3), patch_stride=(2, 1, 3), patch_padding=(0, 0, 0),
            decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=1,
        ),
        'mabe22': dict(
            input_size=(400, 1, 72), in_chans=1, init_embed_dim=96,
            init_num_heads=2, stages=(2, 3, 4), out_embed_dims=(78, 128, 256),
            q_strides=[(2, 1, 4), (2, 1, 6)], mask_unit_attn=(True, False, False),
            patch_kernel=(2, 1, 3), patch_stride=(2, 1, 3), patch_padding=(0, 0, 0),
            decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=1,
        ),
        'calms21': dict(
            # 14 joints x 2D = 28 features → patch_kernel 4 → 7 spatial tokens
            # q_strides: 7 → 7/7=1 → 1/1=1 (temporal only at second stage)
            input_size=(400, 1, 28), in_chans=1, init_embed_dim=96,
            init_num_heads=2, stages=(2, 3, 4), out_embed_dims=(78, 128, 256),
            q_strides=[(2, 1, 7), (2, 1, 1)], mask_unit_attn=(True, False, False),
            patch_kernel=(2, 1, 4), patch_stride=(2, 1, 4), patch_padding=(0, 0, 0),
            decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=1,
        ),
    }

    def __init__(self, model=None, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, dataset: str = 'mabe22',
                        device: str = 'cpu') -> 'BehaveMAE':
        """Load pre-trained BehaveMAE model.

        Args:
            checkpoint_path: path to .pth file
            dataset: config preset ('shot7m2', 'mabe22')
            device: 'cpu' or 'cuda'
        """
        import torch

        config = cls.CONFIGS.get(dataset, cls.CONFIGS['mabe22'])

        try:
            from models.models_defs import hbehavemae
            # Remove keys that hbehavemae() sets explicitly to avoid duplicates
            # hbehavemae() explicitly sets: in_chans, patch_stride (from patch_kernel), patch_padding
            model_config = {k: v for k, v in config.items()
                           if k not in ('in_chans', 'patch_stride', 'patch_padding')}
            model = hbehavemae(**model_config)
        except ImportError:
            raise ImportError(
                "BehaveMAE models not found. Clone and add to PYTHONPATH:\n"
                "  git clone https://github.com/amathislab/BehaveMAE\n"
                "  export PYTHONPATH=$PYTHONPATH:$(pwd)/BehaveMAE"
            )

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)

        return cls(model=model, config=config)

    def encode(self, data: np.ndarray, target_frames: int = 400) -> np.ndarray:
        """Encode skeleton data to feature vector (highest level).

        Uses forward_encoder with mask_ratio=0 (no masking) for clean features.

        Args:
            data: (T, K, D) skeleton coordinates

        Returns:
            (embed_dim,) or (N, embed_dim) feature array
        """
        import torch

        device = next(self.model.parameters()).device
        x = pose_to_behavemae_input(data, target_frames).to(device)
        x = x.unsqueeze(1)  # Add channel dim: (B, T, 1, K*D) -> (B, 1, T, 1, K*D)

        with torch.no_grad():
            latent, _ = self.model.forward_encoder(x, mask_ratio=0)

        return latent.squeeze().cpu().numpy()

    def encode_hierarchical(self, data: np.ndarray,
                            target_frames: int = 400) -> Dict[str, np.ndarray]:
        """Encode skeleton data to multi-scale features using forward hooks.

        Captures per-stage intermediates from the encoder backbone.

        Args:
            data: (T, K, D) skeleton coordinates

        Returns:
            dict mapping level names to feature arrays
        """
        import torch

        device = next(self.model.parameters()).device
        x = pose_to_behavemae_input(data, target_frames).to(device)
        x = x.unsqueeze(1)

        # Register hooks on each stage to capture intermediates
        intermediates = []
        hooks = []
        for stage in self.model.blocks:
            hook = stage.register_forward_hook(
                lambda mod, inp, out, store=intermediates: store.append(out)
            )
            hooks.append(hook)

        with torch.no_grad():
            self.model.forward_encoder(x, mask_ratio=0)

        # Remove hooks
        for h in hooks:
            h.remove()

        return {
            f'level_{i}': feat.squeeze().cpu().numpy()
            for i, feat in enumerate(intermediates)
            if isinstance(feat, torch.Tensor)
        }
