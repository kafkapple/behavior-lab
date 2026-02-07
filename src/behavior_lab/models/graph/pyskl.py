"""PySKL wrapper: ST-GCN, ST-GCN++, CTR-GCN, PoseConv3D, MS-G3D, AAGCN, DG-STGCN.

Successor to mmskeleton (deprecated). Built on MMAction2.
Reference: Duan et al. (ACM MM 2022), "Revisiting Skeleton-based Action Recognition."
Install: pip install pyskl  (or clone https://github.com/kennymckormick/pyskl)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


# Model presets: (config_name, expected_joints, input_dim)
MODEL_PRESETS = {
    'stgcn': dict(type='STGCN', gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn'),
    'stgcn++': dict(type='STGCN', gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn'),
    'ctrgcn': dict(type='CTRGCN'),
    'aagcn': dict(type='AAGCN'),
    'msg3d': dict(type='MSG3D'),
    'dgstgcn': dict(type='DGSTGCN'),
}

# Graph presets for common skeleton formats
GRAPH_PRESETS = {
    'ntu': dict(layout='nturgb+d', mode='spatial'),
    'coco': dict(layout='coco', mode='spatial'),
    'openpose': dict(layout='openpose', mode='spatial'),
}


def pose_to_pyskl_format(
    data: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    num_persons: int = 1,
) -> Dict:
    """Convert (T, K, D) skeleton data to PySKL annotation dict.

    Args:
        data: (T, K, D) skeleton coordinates (D=2 or 3)
        confidence: optional (T, K) confidence scores
        num_persons: M dimension (default 1)

    Returns:
        PySKL-compatible annotation dict with 'keypoint' (M, T, V, C)
    """
    T, K, D = data.shape

    # Reshape to (M, T, V, C)
    keypoint = data.reshape(1, T, K, D)
    if num_persons > 1:
        keypoint = np.pad(keypoint, ((0, num_persons - 1), (0, 0), (0, 0), (0, 0)))

    if confidence is not None:
        keypoint_score = confidence.reshape(1, T, K)
        if num_persons > 1:
            keypoint_score = np.pad(keypoint_score, ((0, num_persons - 1), (0, 0), (0, 0)))
    else:
        keypoint_score = np.ones((num_persons, T, K))

    return {
        'keypoint': keypoint.astype(np.float32),
        'keypoint_score': keypoint_score.astype(np.float32),
        'total_frames': T,
        'img_shape': (1080, 1920),
    }


class PySKLModel:
    """Wrapper for PySKL skeleton-based action recognition models.

    Provides unified interface for ST-GCN family models via pyskl/mmaction2.
    Accepts behavior-lab (T, K, D) format and handles conversion internally.

    Usage:
        model = PySKLModel.from_config('stgcn++', graph='ntu', num_classes=60)
        model = PySKLModel.from_checkpoint('config.py', 'weights.pth')
        probs = model.predict(data)           # (T, K, D) -> (num_classes,)
        features = model.extract_features(data)  # -> (embed_dim,)
    """

    def __init__(self, model=None, cfg=None):
        self.model = model
        self.cfg = cfg
        self._device = 'cpu'

    @classmethod
    def from_checkpoint(cls, config_path: str, checkpoint_path: str,
                        device: str = 'cpu') -> 'PySKLModel':
        """Load model from pyskl config + checkpoint.

        Args:
            config_path: path to pyskl config .py file
            checkpoint_path: path to .pth weights
            device: 'cpu' or 'cuda'
        """
        try:
            from pyskl.apis import init_recognizer
        except ImportError:
            try:
                from mmaction.apis import init_recognizer
            except ImportError:
                raise ImportError(
                    "Install PySKL or MMAction2:\n"
                    "  pip install pyskl\n"
                    "  # or: pip install mmaction2"
                )

        model = init_recognizer(config_path, checkpoint_path, device=device)
        model.eval()

        instance = cls(model=model)
        instance._device = device
        return instance

    @classmethod
    def from_config(cls, model_name: str = 'stgcn++', graph: str = 'ntu',
                    num_classes: int = 60, in_channels: int = 3,
                    device: str = 'cpu', **kwargs) -> 'PySKLModel':
        """Build model from preset config (no pretrained weights).

        Args:
            model_name: 'stgcn', 'stgcn++', 'ctrgcn', 'aagcn', 'msg3d', 'dgstgcn'
            graph: 'ntu', 'coco', 'openpose'
            num_classes: number of action classes
            in_channels: input channels (2 for 2D, 3 for 3D)
            device: 'cpu' or 'cuda'
        """
        import torch
        try:
            from pyskl.models import build_model
            from mmcv import Config
        except ImportError:
            raise ImportError(
                "Install PySKL: pip install pyskl\n"
                "Or clone: https://github.com/kennymckormick/pyskl"
            )

        preset = MODEL_PRESETS.get(model_name.lower(), MODEL_PRESETS['stgcn++'])
        graph_cfg = GRAPH_PRESETS.get(graph.lower(), GRAPH_PRESETS['ntu'])

        cfg = Config(dict(
            model=dict(
                type='RecognizerGCN',
                backbone=dict(**preset, graph_cfg=graph_cfg, in_channels=in_channels),
                cls_head=dict(type='GCNHead', num_classes=num_classes, in_channels=256),
            )
        ))

        model = build_model(cfg.model)
        model.eval()
        model.to(device)

        instance = cls(model=model, cfg=cfg)
        instance._device = device
        return instance

    def predict(self, data: np.ndarray, confidence: Optional[np.ndarray] = None,
                num_persons: int = 1) -> np.ndarray:
        """Predict action class probabilities.

        Args:
            data: (T, K, D) skeleton coordinates
            confidence: optional (T, K) confidence scores
            num_persons: number of persons

        Returns:
            (num_classes,) probability array
        """
        import torch

        anno = pose_to_pyskl_format(data, confidence, num_persons)

        # Build input tensor: (N, M, T, V, C) -> (N, C, T, V, M) for model
        kp = anno['keypoint']  # (M, T, V, C)
        M, T, V, C = kp.shape
        tensor = torch.from_numpy(kp).float()
        tensor = tensor.permute(3, 1, 2, 0)  # (C, T, V, M)
        tensor = tensor.unsqueeze(0).to(self._device)  # (1, C, T, V, M)

        with torch.no_grad():
            # Try pyskl inference API first
            try:
                from pyskl.apis import inference_recognizer
                results = inference_recognizer(self.model, anno)
                if isinstance(results, list):
                    probs = np.zeros(max(r[0] for r in results) + 1)
                    for idx, score in results:
                        probs[idx] = score
                    return probs
            except Exception:
                pass

            # Direct forward pass fallback
            output = self.model(tensor, return_loss=False)
            if isinstance(output, (list, tuple)):
                output = output[0]
            if isinstance(output, torch.Tensor):
                return torch.softmax(output, dim=-1).squeeze().cpu().numpy()
            return np.array(output)

    def extract_features(self, data: np.ndarray,
                         confidence: Optional[np.ndarray] = None,
                         num_persons: int = 1) -> np.ndarray:
        """Extract backbone features (before classification head).

        Args:
            data: (T, K, D) skeleton coordinates

        Returns:
            (embed_dim,) feature vector
        """
        import torch

        anno = pose_to_pyskl_format(data, confidence, num_persons)
        kp = anno['keypoint']  # (M, T, V, C)
        tensor = torch.from_numpy(kp).float()
        tensor = tensor.permute(3, 1, 2, 0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            # Extract from backbone only
            if hasattr(self.model, 'backbone'):
                feat = self.model.backbone(tensor)
                if isinstance(feat, (tuple, list)):
                    feat = feat[-1]
                # Global average pooling
                feat = feat.mean(dim=[2, 3, 4]) if feat.dim() == 5 else feat.mean(dim=[2, 3])
                return feat.squeeze().cpu().numpy()

            # Fallback: full forward
            output = self.model(tensor, return_loss=False)
            if isinstance(output, torch.Tensor):
                return output.squeeze().cpu().numpy()
            return np.array(output).flatten()
