"""Keypoint-MoSeq wrapper: AR-HMM / SLDS behavioral syllable extraction.

Reference: Weinreb et al. (2024), Nature Methods.
Install: pip install keypoint-moseq
"""
import pickle
from pathlib import Path

import numpy as np
from typing import Dict, List, Optional
from sklearn.decomposition import PCA

from ...core.types import ClusteringResult


class KeypointMoSeq:
    """Thin wrapper around keypoint-moseq for behavior-lab (T, K, D) format.

    Keypoint-MoSeq uses switching linear dynamical systems (SLDS) to discover
    behavioral syllables with explicit temporal dynamics modeling.

    Usage:
        model = KeypointMoSeq(project_dir='./moseq_output')
        model.fit(keypoints)              # (T, K, D)
        syllables = model.predict(data)   # (T,) syllable IDs
    """

    def __init__(self, project_dir: str = './moseq_output',
                 num_iters: int = 50, latent_dim: int = 10,
                 kappa: float = 1e6, bodypart_names: Optional[List[str]] = None):
        self.project_dir = project_dir
        self.num_iters = num_iters
        self.latent_dim = latent_dim
        self.kappa = kappa
        self.bodypart_names = bodypart_names
        self._model = None
        self._pca = None
        self._config = {
            'latent_dim': latent_dim,
            'kappa': kappa,
            'conf_pseudocount': 0.001,
            'added_noise_level': 0.1,
        }

    def _to_kpms_format(self, keypoints: np.ndarray, name: str = 'rec_0'):
        """Convert (T, K, D) -> keypoint-moseq dict format."""
        coords = {name: keypoints}
        confs = {name: np.ones(keypoints.shape[:2])}
        return coords, confs

    def fit(self, keypoints: np.ndarray, recording_name: str = 'rec_0') -> 'KeypointMoSeq':
        """Fit AR-HMM + SLDS model on keypoint data.

        Args:
            keypoints: (T, K, D) skeleton coordinates
            recording_name: name for this recording
        """
        try:
            import keypoint_moseq as kpms
        except ImportError:
            raise ImportError("Install keypoint-moseq: pip install keypoint-moseq")

        coords, confs = self._to_kpms_format(keypoints, recording_name)
        bodyparts = self.bodypart_names or [f'kp{i}' for i in range(keypoints.shape[1])]

        # Format data
        data, metadata = kpms.format_data(coords, confs, bodyparts=bodyparts, **self._config)

        # Fit PCA + init model
        self._pca = kpms.fit_pca(data, **self._config)
        model = kpms.init_model(data, pca=self._pca, **self._config)

        # Stage 1: AR-HMM
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir=self.project_dir,
            ar_only=True, num_iters=self.num_iters)

        # Stage 2: Full SLDS
        self._model, self._model_name = kpms.fit_model(
            model, data, metadata, project_dir=self.project_dir,
            ar_only=False, num_iters=self.num_iters)

        self._metadata = metadata
        return self

    def predict(self, keypoints: np.ndarray, recording_name: str = 'new') -> np.ndarray:
        """Extract syllable labels from new data.

        Args:
            keypoints: (T, K, D) skeleton coordinates

        Returns:
            (T,) syllable IDs
        """
        if self._model is None:
            raise RuntimeError("Call .fit() first")

        import keypoint_moseq as kpms

        coords, confs = self._to_kpms_format(keypoints, recording_name)
        bodyparts = self.bodypart_names or [f'kp{i}' for i in range(keypoints.shape[1])]
        data, metadata = kpms.format_data(coords, confs, bodyparts=bodyparts, **self._config)

        results = kpms.apply_model(self._model, data, metadata,
                                   save_results=False, return_model=False)
        return results[recording_name]['syllable']

    def get_results(self, keypoints: np.ndarray, recording_name: str = 'rec') -> Dict:
        """Get full results including latent state and centroid.

        Returns:
            dict with 'syllable', 'latent_state', 'centroid', 'heading'
        """
        if self._model is None:
            raise RuntimeError("Call .fit() first")

        import keypoint_moseq as kpms

        coords, confs = self._to_kpms_format(keypoints, recording_name)
        bodyparts = self.bodypart_names or [f'kp{i}' for i in range(keypoints.shape[1])]
        data, metadata = kpms.format_data(coords, confs, bodyparts=bodyparts, **self._config)

        results = kpms.apply_model(self._model, data, metadata,
                                   save_results=False, return_model=False)
        return results[recording_name]

    def fit_predict(self, keypoints: np.ndarray) -> ClusteringResult:
        """Fit and return structured ClusteringResult."""
        self.fit(keypoints)
        labels = self.predict(keypoints)
        return ClusteringResult(
            labels=labels,
            n_clusters=len(set(labels)),
            metadata={"algorithm": "moseq", "latent_dim": self.latent_dim},
        )

    def save(self, path: str) -> None:
        """Save model state to file."""
        state = {
            "model": self._model,
            "pca": self._pca,
            "config": self._config,
            "bodypart_names": self.bodypart_names,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load model state from file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._pca = state["pca"]
        self._config = state.get("config", self._config)
        self.bodypart_names = state.get("bodypart_names", self.bodypart_names)


class _PCAHMMFallback:
    """Lightweight PCA + HMM fallback when keypoint-moseq is not installed.

    Uses PCA for dimensionality reduction and HMM for temporal segmentation.
    Requires: pip install hmmlearn
    """

    def __init__(
        self,
        n_components: int = 10,
        n_states: int = 20,
        n_iter: int = 50,
    ):
        self.n_components = n_components
        self.n_states = n_states
        self.n_iter = n_iter
        self._pca: PCA | None = None
        self._hmm = None

    def fit(self, keypoints: np.ndarray) -> "ClusteringResult":
        """Fit PCA + HMM on (T, K, D) keypoint data."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("Install hmmlearn: pip install hmmlearn")

        T, K, D = keypoints.shape
        flat = keypoints.reshape(T, K * D)

        self._pca = PCA(n_components=min(self.n_components, flat.shape[1]))
        reduced = self._pca.fit_transform(flat)

        self._hmm = GaussianHMM(
            n_components=self.n_states,
            n_iter=self.n_iter,
            covariance_type="diag",
        )
        self._hmm.fit(reduced)
        labels = self._hmm.predict(reduced)

        return ClusteringResult(
            labels=labels,
            embeddings=reduced[:, :2] if reduced.shape[1] >= 2 else reduced,
            n_clusters=len(set(labels)),
            features=reduced,
            metadata={"algorithm": "pca_hmm_fallback", "n_states": self.n_states},
        )

    def predict(self, keypoints: np.ndarray) -> np.ndarray:
        """Predict syllable labels."""
        if self._hmm is None or self._pca is None:
            raise RuntimeError("Call .fit() first")
        T, K, D = keypoints.shape
        flat = keypoints.reshape(T, K * D)
        reduced = self._pca.transform(flat)
        return self._hmm.predict(reduced)

    def save(self, path: str) -> None:
        state = {"pca": self._pca, "hmm": self._hmm}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._pca = state["pca"]
        self._hmm = state["hmm"]
