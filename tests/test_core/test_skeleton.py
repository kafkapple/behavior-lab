"""Tests for core skeleton module."""
import numpy as np
import pytest

from behavior_lab.core.skeleton import (
    SkeletonDefinition,
    get_skeleton,
    get_skeleton_info,
    list_skeletons,
    register_skeleton,
)


class TestSkeletonDefinition:
    def test_ntu_basic(self):
        s = get_skeleton("ntu")
        assert s.num_joints == 25
        assert s.num_channels == 3
        assert len(s.edges) == 24
        assert len(s.joint_names) == 25

    def test_mars_basic(self):
        s = get_skeleton("mars")
        assert s.num_joints == 7
        assert s.num_channels == 2
        assert s.center_joint == 3
        assert s.num_persons == 2

    def test_dlc_topviewmouse(self):
        s = get_skeleton("dlc_topviewmouse")
        assert s.num_joints == 27
        assert s.num_channels == 2
        assert "nose" in s.joint_names
        assert "tail_end" in s.joint_names

    def test_dlc_quadruped(self):
        s = get_skeleton("quadruped")
        assert s.num_joints == 39
        assert len(s.body_parts) > 0

    def test_adjacency_matrix(self):
        s = get_skeleton("mars")
        A = s.get_adjacency_matrix()
        assert A.shape == (7, 7)
        assert np.all(np.diag(A) == 1)  # self-loops
        assert np.allclose(A, A.T)  # symmetric

    def test_normalized_adjacency(self):
        s = get_skeleton("ntu")
        A_norm = s.get_normalized_adjacency()
        assert A_norm.shape == (25, 25)
        assert not np.any(np.isnan(A_norm))

    def test_inward_outward_edges(self):
        s = get_skeleton("mars")
        inward = s.get_inward_edges()
        outward = s.get_outward_edges()
        assert len(inward) == len(outward)
        # Each inward edge should have a corresponding outward
        for child, parent in inward:
            assert (parent, child) in outward

    def test_get_joint_index(self):
        s = get_skeleton("mars")
        assert s.get_joint_index("nose") == 0
        assert s.get_joint_index("neck") == 3
        with pytest.raises(ValueError):
            s.get_joint_index("nonexistent")

    def test_subset(self):
        s = get_skeleton("dlc_topviewmouse")
        sub = s.subset(["nose", "neck", "mouse_center", "tail_base"])
        assert sub.num_joints == 4
        assert sub.joint_names == ["nose", "neck", "mouse_center", "tail_base"]
        # Edges should only connect joints in the subset
        for i, j in sub.edges:
            assert 0 <= i < 4
            assert 0 <= j < 4

    def test_aliases(self):
        """Different aliases should return the same skeleton."""
        assert get_skeleton("ntu").name == get_skeleton("ntu_rgbd").name
        assert get_skeleton("mars").name == get_skeleton("mars_mouse").name
        assert get_skeleton("dlc_topviewmouse").name == get_skeleton("topviewmouse").name

    def test_unknown_skeleton(self):
        with pytest.raises(ValueError, match="Unknown skeleton"):
            get_skeleton("nonexistent_skeleton")


class TestRegistry:
    def test_list_skeletons(self):
        names = list_skeletons()
        assert len(names) >= 7  # ntu, ucla, coco, mars, calms21, topviewmouse, quadruped

    def test_get_skeleton_info(self):
        info = get_skeleton_info("mars")
        assert info["name"] == "mars_mouse"
        assert info["num_joints"] == 7
        assert "head" in info["body_parts"]

    def test_register_custom(self):
        custom = SkeletonDefinition(
            name="test_custom",
            num_joints=3,
            joint_names=["a", "b", "c"],
            joint_parents=[-1, 0, 1],
            edges=[(0, 1), (1, 2)],
        )
        register_skeleton("test_custom", custom)
        s = get_skeleton("test_custom")
        assert s.num_joints == 3
