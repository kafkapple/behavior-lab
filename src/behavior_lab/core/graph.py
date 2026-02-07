"""
Unified Graph Module - Adjacency matrix construction from SkeletonDefinition.

Provides spatial graph partitioning (Identity, Inward, Outward) required by
GCN-based models (ST-GCN, AGCN, InfoGCN). All graph operations derive from
SkeletonDefinition, eliminating redundant per-dataset Graph classes.
"""
from __future__ import annotations

import numpy as np

from .skeleton import SkeletonDefinition, get_skeleton


def edge2mat(edges: list[tuple[int, int]], num_node: int) -> np.ndarray:
    """Convert edge list to adjacency matrix."""
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    D = np.sum(A, axis=0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if D[i] > 0:
            Dn[i, i] = D[i] ** (-0.5)
    return Dn @ A @ Dn


def get_spatial_graph(skeleton: SkeletonDefinition) -> np.ndarray:
    """Build 3-partition spatial graph: (Identity, Inward, Outward).

    Returns:
        np.ndarray of shape (3, V, V) where V = num_joints.
        - A[0]: Identity (self-loops)
        - A[1]: Normalized inward adjacency
        - A[2]: Normalized outward adjacency
    """
    V = skeleton.num_joints
    self_link = [(i, i) for i in range(V)]
    inward = skeleton.get_inward_edges()
    outward = skeleton.get_outward_edges()

    I = edge2mat(self_link, V)
    In = normalize_adjacency(edge2mat(inward, V))
    Out = normalize_adjacency(edge2mat(outward, V))
    return np.stack((I, In, Out))


class Graph:
    """Unified graph for any skeleton.

    Replaces NTUGraph, UCLAGraph, MARSGraph with a single class that
    derives all adjacency matrices from a SkeletonDefinition.

    Attributes:
        skeleton: The source skeleton definition
        num_node: Number of joints (V)
        A: Spatial partition adjacency (3, V, V)
        A_binary: Symmetric binary adjacency (V, V)
        A_norm: Normalized binary adjacency with self-loops (V, V)
        A_outward_binary: Directed outward adjacency (V, V)
    """

    def __init__(
        self,
        skeleton: SkeletonDefinition | str,
        labeling_mode: str = "spatial",
    ):
        if isinstance(skeleton, str):
            skeleton = get_skeleton(skeleton)

        self.skeleton = skeleton
        self.num_node = skeleton.num_joints

        inward = skeleton.get_inward_edges()
        outward = skeleton.get_outward_edges()
        neighbor = inward + outward

        self.A_binary = edge2mat(neighbor, self.num_node)
        self.A_norm = normalize_adjacency(
            self.A_binary + 2 * np.eye(self.num_node)
        )

        # Directed outward for certain model variants
        A_out = np.zeros((self.num_node, self.num_node))
        for i, j in outward:
            A_out[j, i] = 1
        self.A_outward_binary = A_out

        self.A = self._build(labeling_mode)

    def _build(self, mode: str) -> np.ndarray:
        if mode == "spatial":
            return get_spatial_graph(self.skeleton)
        raise ValueError(f"Unknown labeling mode: {mode}")

    # Convenience properties
    @property
    def joint_names(self) -> list[str]:
        return self.skeleton.joint_names

    @property
    def edges(self) -> list[tuple[int, int]]:
        return self.skeleton.edges


def get_graph(name: str, labeling_mode: str = "spatial") -> Graph:
    """Get a Graph instance by skeleton name.

    Args:
        name: Skeleton name (e.g., 'ntu', 'mars', 'dlc_topviewmouse')
        labeling_mode: Graph partition strategy ('spatial')

    Returns:
        Graph instance with precomputed adjacency matrices.
    """
    return Graph(name, labeling_mode=labeling_mode)
