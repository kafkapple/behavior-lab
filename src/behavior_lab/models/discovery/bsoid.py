"""B-SOiD wrapper: UMAP + HDBSCAN + Random Forest behavioral segmentation.

Reference: Hsu & Yttri (2021), Nature Communications.
Install: pip install umap-learn hdbscan
"""
import numpy as np
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def _compute_bsoid_features(data: np.ndarray, fps: int = 30) -> np.ndarray:
    """Extract B-SOiD spatiotemporal features from (T, K, 2) skeleton data.

    Features per frame:
      - Displacement: ||n_{t+1} - n_t||_2 for each keypoint
      - Pairwise distance: ||n_i - n_j||_2 for keypoint pairs
      - Angular change: arccos of angle between consecutive edge vectors

    Returns:
        (T', n_features) binned to 10fps with smoothing
    """
    T, K, D = data.shape
    assert D == 2, f"B-SOiD requires 2D data, got D={D}"

    # Displacement: (T-1, K)
    disp = np.linalg.norm(np.diff(data, axis=0), axis=-1)

    # Pairwise distances: (T, K*(K-1)/2)
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append(np.linalg.norm(data[:, i] - data[:, j], axis=-1))
    pw_dist = np.stack(pairs, axis=-1)  # (T, n_pairs)

    # Angular change: between consecutive edge vectors
    angles = []
    for i in range(K):
        for j in range(i + 1, K):
            edges = data[:, j] - data[:, i]  # (T, 2)
            # cos(theta) between t and t+1
            dot = np.sum(edges[:-1] * edges[1:], axis=-1)
            norms = np.linalg.norm(edges[:-1], axis=-1) * np.linalg.norm(edges[1:], axis=-1)
            norms = np.clip(norms, 1e-8, None)
            cos_theta = np.clip(dot / norms, -1, 1)
            angles.append(np.arccos(cos_theta))
    ang_change = np.stack(angles, axis=-1)  # (T-1, n_pairs)

    # Align lengths to T-1
    features = np.concatenate([disp, pw_dist[:-1], ang_change], axis=-1)  # (T-1, n_feat)

    # Smoothing: boxcar filter ~60ms
    win = max(1, int(fps * 0.06))
    if win > 1:
        kernel = np.ones(win) / win
        smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, features)
        features = smoothed

    # Temporal binning to 10fps
    bin_size = max(1, fps // 10)
    n_bins = features.shape[0] // bin_size
    features = features[:n_bins * bin_size].reshape(n_bins, bin_size, -1).mean(axis=1)

    return features


class BSOiD:
    """B-SOiD: Behavioral Segmentation via UMAP + HDBSCAN + Random Forest.

    Two-space strategy: UMAP for cluster discovery, RF on high-dim features for prediction.

    Usage:
        model = BSOiD(fps=30)
        results = model.fit(data)  # data: (T, K, 2)
        labels = model.predict(new_data)
    """

    def __init__(self, fps: int = 30, n_neighbors: int = 60, min_dist: float = 0.0,
                 umap_dim: int = 11, min_cluster_size: int = 30, random_state: int = 42):
        self.fps = fps
        self.umap_params = dict(n_neighbors=n_neighbors, min_dist=min_dist,
                                n_components=umap_dim, random_state=random_state)
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.classifier: Optional[RandomForestClassifier] = None

    def fit(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Train B-SOiD on skeleton data.

        Args:
            data: (T, K, 2) pose coordinates

        Returns:
            dict with 'labels', 'embedding_2d', 'n_clusters'
        """
        features = _compute_bsoid_features(data, self.fps)

        self.scaler = StandardScaler()
        features_sc = self.scaler.fit_transform(features)

        # UMAP
        from umap import UMAP
        embeddings = UMAP(**self.umap_params).fit_transform(features_sc)

        # HDBSCAN
        import hdbscan
        labels = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size).fit_predict(embeddings)

        # Random Forest on ORIGINAL high-dim features (not UMAP embeddings)
        self.classifier = RandomForestClassifier(
            n_estimators=200, random_state=self.random_state, n_jobs=-1)
        self.classifier.fit(features_sc, labels)

        return {
            'labels': labels,
            'embedding_2d': embeddings[:, :2] if embeddings.shape[1] >= 2 else embeddings,
            'n_clusters': len(set(labels) - {-1}),
            'features': features_sc,
        }

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict behavior labels using trained RF classifier.

        Args:
            data: (T, K, 2) new pose data

        Returns:
            (T,) per-frame labels (upsampled from 10fps via frameshift)
        """
        if self.classifier is None:
            raise RuntimeError("Call .fit() first")

        features = _compute_bsoid_features(data, self.fps)
        features_sc = self.scaler.transform(features)
        labels_10fps = self.classifier.predict(features_sc)

        # Frameshift upsample to native fps
        bin_size = max(1, self.fps // 10)
        labels = np.repeat(labels_10fps, bin_size)
        # Pad/trim to match original length
        T = data.shape[0]
        if len(labels) < T:
            labels = np.pad(labels, (0, T - len(labels)), mode='edge')
        return labels[:T]
