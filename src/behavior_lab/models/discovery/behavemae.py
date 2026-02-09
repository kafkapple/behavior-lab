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


def pose_to_behavemae_input(data: np.ndarray, target_frames: int = 900,
                            n_agents: int = 3, features_per_agent: int = 24
                            ) -> 'torch.Tensor':
    """Convert skeleton data to BehaveMAE input format (B, 1, T, n_agents, features).

    Supports:
      - (T, n_agents, J, D) e.g. MABe22 (1800, 3, 12, 2)
      - (T, K, D) single-agent fallback

    Args:
        data: skeleton keypoint array
        target_frames: temporal length the model expects
        n_agents: number of agents (spatial dim for model)
        features_per_agent: features per agent (J*D)

    Returns:
        torch.Tensor of shape (1, 1, T, n_agents, features_per_agent)
    """
    import torch

    if data.ndim == 4:
        # (T, n_agents, J, D) → (T, n_agents, J*D)
        T, A, J, D = data.shape
        flat = data.reshape(T, A, J * D).astype(np.float32)
    elif data.ndim == 3:
        # (T, K, D) single-agent → (T, 1, K*D)
        T, K, D = data.shape
        flat = data.reshape(T, 1, K * D).astype(np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D input, got {data.ndim}D")

    # Pad or crop temporal dim to target_frames
    if flat.shape[0] < target_frames:
        flat = np.pad(flat, ((0, target_frames - flat.shape[0]), (0, 0), (0, 0)),
                      mode='edge')
    elif flat.shape[0] > target_frames:
        flat = flat[:target_frames]

    # Pad or crop agent dim
    if flat.shape[1] < n_agents:
        flat = np.pad(flat, ((0, 0), (0, n_agents - flat.shape[1]), (0, 0)),
                      mode='constant')
    elif flat.shape[1] > n_agents:
        flat = flat[:, :n_agents, :]

    # Pad or crop feature dim
    if flat.shape[2] < features_per_agent:
        flat = np.pad(flat, ((0, 0), (0, 0), (0, features_per_agent - flat.shape[2])),
                      mode='constant')
    elif flat.shape[2] > features_per_agent:
        flat = flat[:, :, :features_per_agent]

    # (1, 1, T, n_agents, features_per_agent) — Conv3d expects (B, C, T, H, W)
    tensor = torch.from_numpy(flat).float().unsqueeze(0).unsqueeze(0)
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

    # Default configs per dataset — extracted from checkpoint args
    CONFIGS = {
        'mabe22': dict(
            # Actual checkpoint: input_size=(900, 3, 24), 3 mice × 12 keypoints × 2D
            input_size=(900, 3, 24), in_chans=1, init_embed_dim=128,
            init_num_heads=2, stages=(3, 4, 5), out_embed_dims=(128, 192, 256),
            q_strides=[(5, 1, 1), (1, 3, 1)], mask_unit_attn=(True, False, False),
            patch_kernel=(3, 1, 24), patch_stride=(3, 1, 24), patch_padding=(0, 0, 0),
            decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=1,
            sep_pos_embed=True,
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

    def encode(self, data: np.ndarray, target_frames: int = 900) -> np.ndarray:
        """Encode skeleton data to feature vector (highest level).

        Uses forward_encoder with mask_ratio=0 (no masking) for clean features.

        Args:
            data: (T, n_agents, J, D) or (T, K, D) skeleton coordinates

        Returns:
            (N_tokens, embed_dim) feature array
        """
        import torch

        cfg = self.config
        n_agents = cfg.get('input_size', (900, 3, 24))[1]
        feat_per_agent = cfg.get('input_size', (900, 3, 24))[2]

        device = next(self.model.parameters()).device
        x = pose_to_behavemae_input(
            data, target_frames=target_frames,
            n_agents=n_agents, features_per_agent=feat_per_agent,
        ).to(device)
        # x is already (B, 1, T, n_agents, features) — ready for Conv3d

        with torch.no_grad():
            latent, _ = self.model.forward_encoder(x, mask_ratio=0)

        # latent may have extra size-1 dims, squeeze all then ensure 2D
        out = latent.cpu().numpy().squeeze()
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out  # (N_tokens, embed_dim)

    def encode_hierarchical(self, data: np.ndarray,
                            target_frames: int = 900) -> Dict[str, np.ndarray]:
        """Encode skeleton data to multi-scale features using forward hooks.

        Captures per-stage intermediates from the encoder backbone.

        Args:
            data: (T, n_agents, J, D) or (T, K, D) skeleton coordinates

        Returns:
            dict mapping level names to feature arrays
        """
        import torch

        cfg = self.config
        n_agents = cfg.get('input_size', (900, 3, 24))[1]
        feat_per_agent = cfg.get('input_size', (900, 3, 24))[2]

        device = next(self.model.parameters()).device
        x = pose_to_behavemae_input(
            data, target_frames=target_frames,
            n_agents=n_agents, features_per_agent=feat_per_agent,
        ).to(device)

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
