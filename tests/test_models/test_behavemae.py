"""Tests for BehaveMAE wrapper.

Verifies:
- Input tensor preparation and shape transformations.
- Configuration matching for mabe22.
- Standalone initialization checks.
"""
import os
import numpy as np
import pytest
import torch

from behavior_lab.models.discovery.behavemae import BehaveMAE, pose_to_behavemae_input


class TestBehaveMAEPreprocessing:
    def test_pose_to_behavemae_input_3d(self):
        # 3D shape (T, K, D) e.g., (100, 9, 2)
        data = np.random.randn(100, 9, 2).astype(np.float32)
        tensor = pose_to_behavemae_input(data, target_frames=90, n_agents=1, features_per_agent=18)
        # Expected shape: (1, 1, target_frames, n_agents, features_per_agent) -> (1, 1, 90, 1, 18)
        assert tensor.shape == (1, 1, 90, 1, 18)
        assert isinstance(tensor, torch.Tensor)

    def test_pose_to_behavemae_input_4d(self):
        # 4D shape (T, n_agents, J, D) e.g., (100, 3, 12, 2)
        data = np.random.randn(100, 3, 12, 2).astype(np.float32)
        tensor = pose_to_behavemae_input(data, target_frames=900, n_agents=3, features_per_agent=24)
        assert tensor.shape == (1, 1, 900, 3, 24)

    def test_pose_to_behavemae_input_padding(self):
        # Test temporal padding
        data = np.random.randn(50, 2, 6, 2).astype(np.float32)  # 50 frames
        tensor = pose_to_behavemae_input(data, target_frames=100, n_agents=2, features_per_agent=12)
        assert tensor.shape == (1, 1, 100, 2, 12)

        # Test agent padding
        tensor2 = pose_to_behavemae_input(data, target_frames=50, n_agents=4, features_per_agent=12)
        assert tensor2.shape == (1, 1, 50, 4, 12)


class TestBehaveMAEInitialization:
    def test_default_configs(self):
        assert "mabe22" in BehaveMAE.CONFIGS
        cfg = BehaveMAE.CONFIGS["mabe22"]
        assert cfg["input_size"] == (900, 3, 24)
        assert cfg["decoder_embed_dim"] == 128

    def test_model_not_found_on_invalid_checkpoint(self):
        with pytest.raises((FileNotFoundError, OSError)):
            # hbehavemae might succeed but load should fail for invalid path
            BehaveMAE.from_pretrained("non_existent_checkpoint.pth", dataset="mabe22")

    def test_local_import_possibility(self):
        # hbehavemae is importable if path matches
        try:
            from models.models_defs import hbehavemae
            assert hbehavemae is not None
        except ImportError:
            pytest.fail("external/BehaveMAE models are not importable")
