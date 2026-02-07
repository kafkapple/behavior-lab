"""SUBTLE wrapper: Spectrogram-UMAP-based Temporal Link Embedding.

Reference: Kwon et al. (2024), IJCV.
Install: pip install git+https://github.com/jeakwon/subtle.git
"""
import numpy as np
from typing import Dict, List, Optional


class SUBTLE:
    """Wrapper for SUBTLE library with behavior-lab (T, K, D) format.

    SUBTLE uses Morlet wavelet spectrograms + UMAP embedding + Phenograph clustering
    to discover hierarchical behavioral repertoires (sub/superclusters).

    Usage:
        model = SUBTLE(fps=20)
        results = model.fit([seq1, seq2])  # list of (T_i, K, 3) arrays
        results = model.predict([new_seq])
    """

    def __init__(self, fps: int = 20, n_train_frames: int = 120000,
                 embedding_method: str = 'umap', **kwargs):
        self.fps = fps
        self.n_train_frames = n_train_frames
        self.embedding_method = embedding_method
        self.kwargs = kwargs
        self._mapper = None

    def _preprocess(self, seq: np.ndarray) -> np.ndarray:
        """Convert (T, K, D) -> (T, K*D) with centering."""
        T, K, D = seq.shape
        # Global mean subtraction (SUBTLE's avatar_preprocess)
        mean = seq.mean(axis=(0, 1))
        centered = seq - mean
        return centered.reshape(T, K * D)

    def fit(self, sequences: List[np.ndarray]) -> Dict:
        """Fit SUBTLE model on skeleton sequences.

        Args:
            sequences: list of (T_i, K, 3) 3D skeleton arrays

        Returns:
            dict with 'embeddings', 'subclusters', 'superclusters',
                       'transitions', 'retention'
        """
        try:
            import subtle
        except ImportError:
            raise ImportError(
                "Install SUBTLE: pip install git+https://github.com/jeakwon/subtle.git")

        flat = [self._preprocess(s) for s in sequences]
        self._mapper = subtle.Mapper(
            fs=self.fps, n_train_frames=self.n_train_frames,
            embedding_method=self.embedding_method, **self.kwargs)
        outputs = self._mapper.fit(flat)

        return {
            'embeddings': outputs.get('Z'),
            'subclusters': outputs.get('y'),
            'superclusters': outputs.get('Y'),
            'transitions': outputs.get('TP'),
            'retention': outputs.get('R'),
        }

    def predict(self, sequences: List[np.ndarray]) -> Dict:
        """Apply trained model to new sequences.

        Args:
            sequences: list of (T_i, K, 3) 3D skeleton arrays

        Returns:
            same structure as fit()
        """
        if self._mapper is None:
            raise RuntimeError("Call .fit() first")

        flat = [self._preprocess(s) for s in sequences]
        outputs = self._mapper.run(flat)

        return {
            'embeddings': outputs.get('Z'),
            'subclusters': outputs.get('y'),
            'superclusters': outputs.get('Y'),
            'transitions': outputs.get('TP'),
            'retention': outputs.get('R'),
        }

    def save(self, path: str):
        """Save trained mapper."""
        if self._mapper is None:
            raise RuntimeError("No trained model to save")
        self._mapper.save(path)

    def load(self, path: str):
        """Load pre-trained mapper."""
        import subtle
        self._mapper = subtle.Mapper()
        self._mapper.load(path)
        return self
