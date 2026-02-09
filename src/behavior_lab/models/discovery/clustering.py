"""Unsupervised clustering pipeline: PCA -> UMAP -> KMeans."""
import numpy as np
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def cluster_features(
    features: np.ndarray,
    n_clusters: int = 5,
    pca_variance: float = 0.95,
    use_umap: bool = True,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Run clustering pipeline on feature matrix.

    Args:
        features: (N, D) feature matrix
        n_clusters: number of clusters
        pca_variance: PCA variance ratio to keep
        use_umap: use UMAP for 2D embedding
        random_state: random seed

    Returns:
        dict with 'labels', 'embedding_2d', 'pca_variance_explained'
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # PCA
    pca = PCA(n_components=min(X.shape[0], X.shape[1], 50), random_state=random_state)
    X_pca = pca.fit_transform(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumvar, pca_variance) + 1
    X_pca = X_pca[:, :n_components]

    # UMAP embedding (optional)
    embedding_2d = X_pca[:, :2]  # fallback
    if use_umap:
        try:
            from umap import UMAP
            embedding_2d = UMAP(n_neighbors=15, min_dist=0.1,
                                random_state=random_state).fit_transform(X_pca)
        except ImportError:
            pass

    # KMeans
    labels = KMeans(n_clusters=n_clusters, random_state=random_state,
                    n_init=10).fit_predict(X_pca)

    return {
        'labels': labels,
        'embedding_2d': embedding_2d,
        'pca_variance_explained': cumvar.tolist(),
        'n_clusters': n_clusters,
    }
