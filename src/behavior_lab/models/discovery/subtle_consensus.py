"""PCA-space consensus clustering for SUBTLE (clustering="pca_consensus").

SUBTLE's Mapper clusters the 2D UMAP output (`pheno.fit_predict(Z)`, `pheno.predict(Z)`). This module swaps two Mapper
attributes so the clusters come from the 100-D PCA space instead, while UMAP stays for display:
- `PCKeep` wraps the UMAP and remembers the PCA input of its last call;
- `PCAConsensus` replaces Phenograph: 5 Phenograph-Leiden runs on PCA (seeds 5*seed+0..4), a kNN graph weighted by how often
  two frames share a cluster, one final Leiden whose resolution is bisected until the cluster count is within 10% of what
  Phenograph finds on the UMAP (the original SUBTLE count), then 1-NN on PCA to label any frame.
Mapper calls umap.transform(PC) right before pheno.predict(Z) in fit and run, so `last` is always the matching PCA block.
Evidence (y5a5, 10 sessions x 2 seeds): human-label AMI 0.273 -> 0.266, seed-pair AMI 0.759 -> 0.859
(vault note 261024_SUBTLE_원논문_UMAP_근거_재검토, script subtle_ref/_proposals/cwt_compat/paper_criteria.py).
"""
from __future__ import annotations

import numpy as np

N_RUNS, K, TOL = 5, 30, 0.1


class PCKeep:
    """UMAP proxy that keeps the PCA block it was given (dropped on pickling)."""

    def __init__(self, umap):
        self.umap, self.last = umap, None

    def fit_transform(self, X):
        self.last = X
        return self.umap.fit_transform(X)

    def transform(self, X):
        self.last = X
        return self.umap.transform(X)

    def __getstate__(self):
        return {"umap": self.umap, "last": None}


def _pg(X: np.ndarray, seed: int) -> np.ndarray:
    import phenograph
    return phenograph.cluster(X, k=K, clustering_algo="leiden", min_cluster_size=10, seed=seed, n_jobs=1)[0]


def _leiden(g, w, res: float) -> np.ndarray:
    import leidenalg as la
    return np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights=w, resolution_parameter=res, seed=0).membership)


def consensus_labels(P: np.ndarray, k_target: int, seed: int) -> tuple[np.ndarray, float]:
    """Co-assignment graph of N_RUNS Phenograph runs on P, then Leiden with resolution bisected to k_target (+-TOL)."""
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors
    runs = [_pg(P, seed * N_RUNS + i) for i in range(N_RUNS)]  # seeds 0 and 1 must not share runs (0..4 vs 5..9)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    ii, jj = np.repeat(np.arange(len(P)), K), nn.ravel()
    w = np.mean([r[ii] == r[jj] for r in runs], axis=0)
    keep = w > 0
    g = ig.Graph(n=len(P), edges=list(zip(ii[keep].tolist(), jj[keep].tolist())), directed=False)
    w = w[keep].tolist()
    lo, hi = 0.1, 20.0
    for _ in range(25):
        r = float(np.sqrt(lo * hi))
        lab = _leiden(g, w, r)
        k = len(set(lab))
        if abs(k - k_target) <= TOL * k_target:
            break
        lo, hi = (r, hi) if k < k_target else (lo, r)
    return lab, r


class PCAConsensus:
    """Drop-in for subtle.module.Phenograph: fit_predict(Z) / predict(Z), labels taken from `space.last` (PCA)."""

    def __init__(self, space: PCKeep, seed: int = 0):
        self.space, self.seed = space, seed

    def fit_predict(self, Z, y=None):
        from sklearn.neighbors import KNeighborsClassifier
        P = self.space.last
        assert P is not None and len(P) == len(Z), "PCKeep must see the PCA block right before clustering"
        self.k_umap = len(set(_pg(Z, self.seed)))  # the cluster count original SUBTLE would give
        lab, self.resolution = consensus_labels(P, self.k_umap, self.seed)
        self.neigh = KNeighborsClassifier(n_neighbors=1).fit(P, lab)
        self.labels = np.unique(lab)
        return lab

    def predict(self, Z):
        P = self.space.last
        assert P is not None and len(P) == len(Z), "PCKeep must see the PCA block right before clustering"
        return self.neigh.predict(P)
