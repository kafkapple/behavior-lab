"""Tests for tensor format conversion."""
import numpy as np
import pytest

from behavior_lab.core.skeleton import get_skeleton
from behavior_lab.core.tensor_format import graph_to_sequence, sequence_to_graph


class TestSequenceToGraph:
    def test_single_person_3d(self):
        """(T, K, D) -> (C, T, V, M)"""
        s = get_skeleton("ntu")
        seq = np.random.randn(64, 25, 3)
        tensor = sequence_to_graph(seq, s, num_persons=1)
        assert tensor.shape == (3, 64, 25, 1)

    def test_multi_person_2d(self):
        """(T, K, D) for M=2 -> (C, T, V, M) with padding."""
        s = get_skeleton("mars")
        seq = np.random.randn(64, 7, 2)
        tensor = sequence_to_graph(seq, s)  # M=2 from skeleton
        assert tensor.shape == (2, 64, 7, 2)

    def test_multi_person_flattened(self):
        """(T, M*K, D) -> (C, T, V, M)"""
        s = get_skeleton("mars")
        seq = np.random.randn(64, 14, 2)  # 2*7 = 14
        tensor = sequence_to_graph(seq, s)
        assert tensor.shape == (2, 64, 7, 2)

    def test_batched(self):
        """(N, T, K, D) -> (N, C, T, V, M)"""
        s = get_skeleton("ntu")
        seq = np.random.randn(8, 64, 25, 3)
        tensor = sequence_to_graph(seq, s, num_persons=2)
        assert tensor.shape == (8, 3, 64, 25, 2)

    def test_max_frames_pad(self):
        s = get_skeleton("mars")
        seq = np.random.randn(30, 7, 2)
        tensor = sequence_to_graph(seq, s, num_persons=1, max_frames=64)
        assert tensor.shape == (2, 64, 7, 1)

    def test_max_frames_crop(self):
        s = get_skeleton("mars")
        seq = np.random.randn(100, 7, 2)
        tensor = sequence_to_graph(seq, s, num_persons=1, max_frames=64)
        assert tensor.shape == (2, 64, 7, 1)

    def test_channel_padding(self):
        """2D data for a 3D skeleton should be zero-padded."""
        s = get_skeleton("ntu")  # 3D skeleton
        seq = np.random.randn(64, 25, 2)  # 2D input
        tensor = sequence_to_graph(seq, s, num_persons=1)
        assert tensor.shape == (3, 64, 25, 1)
        # Third channel should be zero
        assert np.allclose(tensor[2], 0)


class TestGraphToSequence:
    def test_single_person(self):
        """(C, T, V, M=1) -> (T, V, C)"""
        tensor = np.random.randn(3, 64, 25, 1)
        seq = graph_to_sequence(tensor)
        assert seq.shape == (64, 25, 3)

    def test_multi_person(self):
        """(C, T, V, M=2) -> (T, M*V, C)"""
        tensor = np.random.randn(2, 64, 7, 2)
        seq = graph_to_sequence(tensor)
        assert seq.shape == (64, 14, 2)

    def test_batched(self):
        """(N, C, T, V, M=1) -> (N, T, V, C)"""
        tensor = np.random.randn(8, 3, 64, 25, 1)
        seq = graph_to_sequence(tensor)
        assert seq.shape == (8, 64, 25, 3)


class TestRoundTrip:
    def test_single_person_roundtrip(self):
        """seq -> graph -> seq should preserve data."""
        s = get_skeleton("mars")
        original = np.random.randn(64, 7, 2)
        tensor = sequence_to_graph(original, s, num_persons=1)
        recovered = graph_to_sequence(tensor)
        np.testing.assert_allclose(original, recovered, atol=1e-10)

    def test_batched_roundtrip(self):
        s = get_skeleton("ntu")
        original = np.random.randn(4, 64, 25, 3)
        tensor = sequence_to_graph(original, s, num_persons=1)
        recovered = graph_to_sequence(tensor)
        np.testing.assert_allclose(original, recovered, atol=1e-10)
