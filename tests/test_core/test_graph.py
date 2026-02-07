"""Tests for core graph module."""
import numpy as np

from behavior_lab.core.graph import Graph, get_graph


class TestGraph:
    def test_ntu_graph(self):
        g = get_graph("ntu")
        assert g.num_node == 25
        assert g.A.shape == (3, 25, 25)
        assert g.A_binary.shape == (25, 25)
        assert g.A_norm.shape == (25, 25)

    def test_mars_graph(self):
        g = get_graph("mars")
        assert g.num_node == 7
        assert g.A.shape == (3, 7, 7)

    def test_dlc_topviewmouse_graph(self):
        g = get_graph("dlc_topviewmouse")
        assert g.num_node == 27
        assert g.A.shape == (3, 27, 27)

    def test_spatial_partition(self):
        """Identity + Inward + Outward partitions."""
        g = get_graph("mars")
        I, In, Out = g.A[0], g.A[1], g.A[2]
        # Identity should be diagonal
        assert np.allclose(I, np.eye(7))

    def test_convenience_properties(self):
        g = get_graph("mars")
        assert len(g.joint_names) == 7
        assert len(g.edges) == 7

    def test_from_skeleton_object(self):
        from behavior_lab.core.skeleton import get_skeleton
        skel = get_skeleton("ucla")
        g = Graph(skel)
        assert g.num_node == 20
