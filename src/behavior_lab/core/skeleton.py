"""
Skeleton Registry - Centralized keypoint and skeleton definitions.

Provides a unified registry for all skeleton types used in pose estimation
and action recognition. Supports both programmatic and YAML-based definitions.

Supported Skeletons:
- Human: NTU RGB+D (25 joints), NW-UCLA (20 joints), COCO (17 joints)
- Animal: MARS Mouse (7 joints), CalMS21 Mouse (7 joints)
- DLC: TopViewMouse (27 joints), Quadruped (39 joints)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SkeletonDefinition:
    """Complete skeleton definition with keypoints and connections.

    Attributes:
        name: Unique identifier (e.g., 'ntu_rgbd', 'mars_mouse')
        num_joints: Number of keypoints
        joint_names: Ordered list of joint names
        joint_parents: Parent index per joint (-1 for root)
        edges: (parent, child) tuples defining the skeleton graph
        symmetric_pairs: Left-right joint index pairs for augmentation
        num_channels: Coordinate dimensions (2 for xy, 3 for xyz)
        coordinate_system: 'xy' or 'xyz'
        body_parts: Named groups of joint indices
        center_joint: Index of the center joint for normalization
        num_persons: Default number of persons/animals in the scene
        joint_colors: Optional per-joint colors for visualization
        limb_colors: Optional per-limb colors for visualization
    """
    name: str
    num_joints: int
    joint_names: list[str]
    joint_parents: list[int]
    edges: list[tuple[int, int]]

    symmetric_pairs: list[tuple[int, int]] = field(default_factory=list)
    num_channels: int = 3
    coordinate_system: str = "xyz"
    body_parts: dict[str, list[int]] = field(default_factory=dict)
    center_joint: int = 0
    num_persons: int = 1

    joint_colors: list[str] | None = None
    limb_colors: list[str] | None = None

    def get_adjacency_matrix(self) -> np.ndarray:
        """Binary adjacency matrix with self-loops. Shape: (V, V)."""
        A = np.zeros((self.num_joints, self.num_joints))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        np.fill_diagonal(A, 1)
        return A

    def get_normalized_adjacency(self) -> np.ndarray:
        """Degree-normalized adjacency: D^{-1/2} A D^{-1/2}."""
        A = self.get_adjacency_matrix()
        D = np.diag(np.sum(A, axis=1) ** (-0.5))
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return D @ A @ D

    def get_inward_edges(self) -> list[tuple[int, int]]:
        """Edges pointing toward root (child → parent)."""
        return [
            (j, self.joint_parents[j])
            for j in range(self.num_joints)
            if self.joint_parents[j] >= 0
        ]

    def get_outward_edges(self) -> list[tuple[int, int]]:
        """Edges pointing away from root (parent → child)."""
        return [
            (self.joint_parents[j], j)
            for j in range(self.num_joints)
            if self.joint_parents[j] >= 0
        ]

    def get_joint_index(self, name: str) -> int:
        """Get joint index by name. Raises ValueError if not found."""
        try:
            return self.joint_names.index(name)
        except ValueError:
            raise ValueError(
                f"Joint '{name}' not found in skeleton '{self.name}'. "
                f"Available: {self.joint_names}"
            )

    def subset(self, joint_names: list[str]) -> "SkeletonDefinition":
        """Create a sub-skeleton with only the specified joints.

        Useful for keypoint preset filtering (e.g., MARS 7 from TopViewMouse 27).
        """
        indices = [self.get_joint_index(n) for n in joint_names]
        idx_map = {old: new for new, old in enumerate(indices)}

        new_edges = []
        for i, j in self.edges:
            if i in idx_map and j in idx_map:
                new_edges.append((idx_map[i], idx_map[j]))

        new_parents = []
        for old_idx in indices:
            parent = self.joint_parents[old_idx]
            new_parents.append(idx_map.get(parent, -1))

        new_symmetric = []
        for i, j in self.symmetric_pairs:
            if i in idx_map and j in idx_map:
                new_symmetric.append((idx_map[i], idx_map[j]))

        new_body_parts = {}
        for part_name, part_indices in self.body_parts.items():
            mapped = [idx_map[i] for i in part_indices if i in idx_map]
            if mapped:
                new_body_parts[part_name] = mapped

        new_colors = None
        if self.joint_colors:
            new_colors = [self.joint_colors[i] for i in indices]

        return SkeletonDefinition(
            name=f"{self.name}_subset{len(indices)}",
            num_joints=len(indices),
            joint_names=joint_names,
            joint_parents=new_parents,
            edges=new_edges,
            symmetric_pairs=new_symmetric,
            num_channels=self.num_channels,
            coordinate_system=self.coordinate_system,
            body_parts=new_body_parts,
            center_joint=idx_map.get(self.center_joint, 0),
            num_persons=self.num_persons,
            joint_colors=new_colors,
        )

    def __repr__(self) -> str:
        return (
            f"SkeletonDefinition(name='{self.name}', "
            f"joints={self.num_joints}, edges={len(self.edges)}, "
            f"channels={self.num_channels})"
        )


# =============================================================================
# Human Skeleton Definitions
# =============================================================================

NTU_SKELETON = SkeletonDefinition(
    name="ntu_rgbd",
    num_joints=25,
    # 25 joints from Kinect V2 body tracking (0-indexed)
    # Coordinate system: Y-up (Kinect: X=horizontal, Y=vertical, Z=depth)
    # Edges match ST-GCN (Yan et al., AAAI 2018) ntu_rgb_d graph definition
    # Reference: Shahroudy et al., "NTU RGB+D" (CVPR 2016), 60 action classes
    #            Liu et al., "NTU RGB+D 120" (TPAMI 2020), 120 action classes
    joint_names=[
        "base_spine",      # 0
        "mid_spine",       # 1
        "neck",            # 2
        "head",            # 3
        "left_shoulder",   # 4
        "left_elbow",      # 5
        "left_wrist",      # 6
        "left_hand",       # 7
        "right_shoulder",  # 8
        "right_elbow",     # 9
        "right_wrist",     # 10
        "right_hand",      # 11
        "left_hip",        # 12
        "left_knee",       # 13
        "left_ankle",      # 14
        "left_foot",       # 15
        "right_hip",       # 16
        "right_knee",      # 17
        "right_ankle",     # 18
        "right_foot",      # 19
        "spine",           # 20 (shoulder center / SpineShoulder)
        "left_hand_tip",   # 21 (HandTipLeft)
        "left_thumb",      # 22 (ThumbLeft)
        "right_hand_tip",  # 23 (HandTipRight)
        "right_thumb",     # 24 (ThumbRight)
    ],
    joint_parents=[
        -1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10,
        0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
    ],
    edges=[
        (0, 1), (1, 20), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19),
    ],
    symmetric_pairs=[
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (21, 23), (22, 24),
    ],
    num_channels=3,
    coordinate_system="xyz",
    body_parts={
        "torso": [0, 1, 2, 20],
        "head": [2, 3],
        "left_arm": [4, 5, 6, 7, 21, 22],
        "right_arm": [8, 9, 10, 11, 23, 24],
        "left_leg": [12, 13, 14, 15],
        "right_leg": [16, 17, 18, 19],
    },
    center_joint=0,
    num_persons=2,
)


UCLA_SKELETON = SkeletonDefinition(
    name="nw_ucla",
    num_joints=20,
    # 20 joints from Kinect V1 body tracking (0-indexed)
    # Coordinate system: Y-up (Kinect SDK v1)
    # Reference: Wang et al., "Cross-View Action Modeling" (CVPR 2014)
    #            10 action classes, 3 camera views
    joint_names=[
        "hip_center",       # 0
        "spine",            # 1
        "shoulder_center",  # 2
        "head",             # 3
        "left_shoulder",    # 4
        "left_elbow",       # 5
        "left_wrist",       # 6
        "left_hand",        # 7
        "right_shoulder",   # 8
        "right_elbow",      # 9
        "right_wrist",      # 10
        "right_hand",       # 11
        "left_hip",         # 12
        "left_knee",        # 13
        "left_ankle",       # 14
        "left_foot",        # 15
        "right_hip",        # 16
        "right_knee",       # 17
        "right_ankle",      # 18
        "right_foot",       # 19
    ],
    joint_parents=[
        -1, 0, 1, 2, 2, 4, 5, 6, 2, 8, 9, 10,
        0, 12, 13, 14, 0, 16, 17, 18,
    ],
    edges=[
        (0, 1), (1, 2), (2, 3),
        (2, 4), (4, 5), (5, 6), (6, 7),
        (2, 8), (8, 9), (9, 10), (10, 11),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19),
    ],
    symmetric_pairs=[
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
    ],
    num_channels=3,
    coordinate_system="xyz",
    body_parts={
        "torso": [0, 1, 2],
        "head": [3],
        "left_arm": [4, 5, 6, 7],
        "right_arm": [8, 9, 10, 11],
        "left_leg": [12, 13, 14, 15],
        "right_leg": [16, 17, 18, 19],
    },
    center_joint=0,
)


COCO_SKELETON = SkeletonDefinition(
    name="coco",
    num_joints=17,
    joint_names=[
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ],
    joint_parents=[
        -1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14,
    ],
    edges=[
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ],
    symmetric_pairs=[
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16),
    ],
    num_channels=2,
    coordinate_system="xy",
    body_parts={
        "head": [0, 1, 2, 3, 4],
        "torso": [5, 6, 11, 12],
        "left_arm": [5, 7, 9],
        "right_arm": [6, 8, 10],
        "left_leg": [11, 13, 15],
        "right_leg": [12, 14, 16],
    },
    center_joint=0,
)


# =============================================================================
# Animal Skeleton Definitions
# =============================================================================

MARS_MOUSE_SKELETON = SkeletonDefinition(
    name="mars_mouse",
    num_joints=7,
    # 7 keypoints from MARS top-view mouse pose estimation
    # Coordinate system: 2D pixel (top-view camera)
    # Reference: Segalin et al., "MARS" (eLife 2021)
    #            Top-view: nose, left_ear, right_ear, neck, left_hip, right_hip, tail_base
    joint_names=[
        "nose", "left_ear", "right_ear", "neck",
        "left_hip", "right_hip", "tail_base",
    ],
    joint_parents=[3, 3, 3, -1, 3, 3, 4],
    edges=[
        (0, 3), (1, 3), (2, 3),
        (3, 4), (3, 5), (4, 6), (5, 6),
    ],
    symmetric_pairs=[(1, 2), (4, 5)],
    num_channels=2,
    coordinate_system="xy",
    body_parts={
        "head": [0, 1, 2, 3],
        "body": [3, 4, 5],
        "tail": [6],
    },
    center_joint=3,
    num_persons=2,
)


CALMS21_MOUSE_SKELETON = SkeletonDefinition(
    name="calms21_mouse",
    num_joints=7,
    # 7 keypoints, same as MARS (resident-intruder dyad assay)
    # Coordinate system: 2D pixel (top-view camera)
    # Data shape: (N, 2, T, 7, 2) — 2 mice, T frames, 7 joints, xy
    # Reference: Sun et al., "CalMS21" (NeurIPS 2021 Datasets & Benchmarks)
    #            6M+ frames, 1M+ annotated, behavior classification
    joint_names=[
        "nose", "left_ear", "right_ear", "neck",
        "left_hip", "right_hip", "tail_base",
    ],
    joint_parents=[3, 3, 3, -1, 3, 3, 4],
    edges=[
        (0, 3), (1, 3), (2, 3),
        (3, 4), (3, 5), (4, 6), (5, 6),
    ],
    symmetric_pairs=[(1, 2), (4, 5)],
    num_channels=2,
    coordinate_system="xy",
    body_parts={
        "head": [0, 1, 2, 3],
        "body": [3, 4, 5],
        "tail": [6],
    },
    center_joint=3,
    num_persons=2,
)


# =============================================================================
# 3D Rodent Skeleton Definitions
# =============================================================================

RAT7M_SKELETON = SkeletonDefinition(
    name="rat7m",
    num_joints=20,
    # 20 joints from DANNCE 3D markerless motion capture
    # Coordinate system: Z-up (3D motion capture arena)
    # Reference: Dunn et al., "Geometric deep learning on 3D animal pose" (2021)
    #            Figshare collection 5295370, ~7M frames
    joint_names=[
        "nose_tip", "head_top", "left_ear", "right_ear",
        "neck", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "spine_mid",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "tail_base", "tail_mid",
    ],
    joint_parents=[
        4, 4, 4, 4,
        -1,
        4, 4,
        5, 6,
        7, 8,
        4,
        11, 11,
        12, 13,
        14, 15,
        11, 18,
    ],
    edges=[
        (0, 4), (1, 4), (2, 4), (3, 4),
        (4, 5), (4, 6),
        (5, 7), (6, 8),
        (7, 9), (8, 10),
        (4, 11),
        (11, 12), (11, 13),
        (12, 14), (13, 15),
        (14, 16), (15, 17),
        (11, 18), (18, 19),
    ],
    symmetric_pairs=[
        (2, 3), (5, 6), (7, 8), (9, 10),
        (12, 13), (14, 15), (16, 17),
    ],
    num_channels=3,
    coordinate_system="xyz",
    body_parts={
        "head": [0, 1, 2, 3],
        "torso": [4, 5, 6, 11],
        "left_arm": [5, 7, 9],
        "right_arm": [6, 8, 10],
        "left_leg": [12, 14, 16],
        "right_leg": [13, 15, 17],
        "tail": [18, 19],
    },
    center_joint=4,
)


SUBTLE_MOUSE_SKELETON = SkeletonDefinition(
    name="subtle_mouse",
    num_joints=9,
    # Column order from SUBTLE kinematics.py avatar_configs['nodes']:
    #   nose=0, neck=1, anus=2, chest=3, rfoot=4, lfoot=5, rhand=6, lhand=7, tip=8
    # Edge topology (kinematics.py avatar_configs['edges']):
    #   head=[0,1], fbody=[1,3], hbody=[3,2], rleg=[4,2], lleg=[5,2],
    #   rarm=[6,3], larm=[7,3], tail=[2,8]
    # Body axis: nose→neck→chest(mid_back)→anus(tail_base)→tip(tail_tip)
    # Coordinate system: Z-up (3D motion capture), sampling rate: 20 Hz
    # Reference: Kwon et al., "SUBTLE" (github.com/jeakwon/subtle)
    joint_names=[
        "nose", "neck", "tail_base", "mid_back", "right_hindpaw",
        "left_hindpaw", "right_forepaw", "left_forepaw", "tail_tip",
    ],
    joint_parents=[1, 3, 3, -1, 2, 2, 3, 3, 2],
    edges=[
        (0, 1),  # head: nose→neck
        (1, 3),  # front body: neck→mid_back (chest)
        (3, 2),  # hind body: mid_back→tail_base (anus)
        (7, 3),  # left forepaw→mid_back
        (6, 3),  # right forepaw→mid_back
        (5, 2),  # left hindpaw→tail_base
        (4, 2),  # right hindpaw→tail_base
        (2, 8),  # tail: tail_base→tail_tip
    ],
    symmetric_pairs=[(4, 5), (6, 7)],
    num_channels=3,
    coordinate_system="xyz",
    body_parts={
        "head": [0, 1],
        "body": [2, 3],
        "left_front_leg": [7],
        "right_front_leg": [6],
        "left_back_leg": [5],
        "right_back_leg": [4],
        "tail": [8],
    },
    center_joint=3,
)


MABE22_MOUSE_SKELETON = SkeletonDefinition(
    name="mabe22_mouse",
    num_joints=12,
    # 12 keypoints per mouse, 3 mice in triplet configuration
    # Coordinate system: 2D pixel (top-view camera)
    # Data shape: (N, T, 36, 2) where 36 = 3 mice × 12 joints
    # Reference: Sun et al., "MABe22" (ICML 2023)
    #            Multi-species multi-task benchmark
    joint_names=[
        "nose",              # 0
        "left_ear",          # 1
        "right_ear",         # 2
        "neck",              # 3
        "left_forepaw",      # 4
        "right_forepaw",     # 5
        "center_back",       # 6
        "left_hindpaw",      # 7
        "right_hindpaw",     # 8
        "tail_base",         # 9
        "tail_middle",       # 10
        "tail_tip",          # 11
    ],
    joint_parents=[3, 3, 3, -1, 6, 6, 3, 6, 6, 6, 9, 10],
    edges=[
        (0, 3), (1, 3), (2, 3), (3, 6),
        (6, 4), (6, 5), (6, 7), (6, 8),
        (6, 9), (9, 10), (10, 11),
    ],
    symmetric_pairs=[(1, 2), (4, 5), (7, 8)],
    num_channels=2,
    coordinate_system="xy",
    body_parts={
        "head": [0, 1, 2, 3],
        "body": [6],
        "front_paws": [4, 5],
        "hind_paws": [7, 8],
        "tail": [9, 10, 11],
    },
    center_joint=6,
    num_persons=3,
)


SHANK3KO_MOUSE_SKELETON = SkeletonDefinition(
    name="shank3ko_mouse",
    num_joints=16,
    # Joint names from raw .mat Body_name field (Shank3KO_mice_slk3D.mat)
    # Coordinate system: Z-up (3D SLEAP tracking), sampling rate: 30 Hz
    # Reference: Zenodo dataset, 16 joints, single mouse
    joint_names=[
        "nose", "left_ear", "right_ear", "neck",
        "left_front_limb", "right_front_limb",
        "left_hind_limb", "right_hind_limb",
        "left_front_claw", "right_front_claw",
        "left_hind_claw", "right_hind_claw",
        "back", "root_tail", "mid_tail", "tip_tail",
    ],
    joint_parents=[
        3, 3, 3, -1,    # nose, ears -> neck
        12, 12, 12, 12,  # limbs -> back
        4, 5, 6, 7,      # claws -> limbs
        3, 12, 13, 14,   # back -> neck, tail chain
    ],
    edges=[
        (0, 3), (1, 3), (2, 3), (3, 12),
        (12, 4), (12, 5), (12, 6), (12, 7),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 13), (13, 14), (14, 15),
    ],
    symmetric_pairs=[
        (1, 2), (4, 5), (6, 7), (8, 9), (10, 11),
    ],
    num_channels=3,
    coordinate_system="xyz",
    body_parts={
        "head": [0, 1, 2, 3],
        "body": [12],
        "left_front_leg": [4, 8],
        "right_front_leg": [5, 9],
        "left_back_leg": [6, 10],
        "right_back_leg": [7, 11],
        "tail": [13, 14, 15],
    },
    center_joint=3,
)


# =============================================================================
# DLC SuperAnimal Skeleton Definitions
# =============================================================================

DLC_TOPVIEWMOUSE_SKELETON = SkeletonDefinition(
    name="dlc_topviewmouse",
    num_joints=27,
    joint_names=[
        "nose",             # 0
        "left_ear",         # 1
        "right_ear",        # 2
        "left_ear_tip",     # 3
        "right_ear_tip",    # 4
        "left_eye",         # 5
        "right_eye",        # 6
        "neck",             # 7
        "mid_back",         # 8
        "mouse_center",     # 9
        "mid_backend",      # 10
        "mid_backend2",     # 11
        "mid_backend3",     # 12
        "tail_base",        # 13
        "tail1",            # 14
        "tail2",            # 15
        "tail3",            # 16
        "tail4",            # 17
        "tail5",            # 18
        "left_shoulder",    # 19
        "left_midside",     # 20
        "left_hip",         # 21
        "right_shoulder",   # 22
        "right_midside",    # 23
        "right_hip",        # 24
        "tail_end",         # 25
        "head_midpoint",    # 26
    ],
    joint_parents=[
        26,  # 0: nose -> head_midpoint
        26,  # 1: left_ear -> head_midpoint
        26,  # 2: right_ear -> head_midpoint
        1,   # 3: left_ear_tip -> left_ear
        2,   # 4: right_ear_tip -> right_ear
        26,  # 5: left_eye -> head_midpoint
        26,  # 6: right_eye -> head_midpoint
        26,  # 7: neck -> head_midpoint
        7,   # 8: mid_back -> neck
        8,   # 9: mouse_center -> mid_back
        9,   # 10: mid_backend -> mouse_center
        10,  # 11: mid_backend2 -> mid_backend
        11,  # 12: mid_backend3 -> mid_backend2
        12,  # 13: tail_base -> mid_backend3
        13,  # 14: tail1 -> tail_base
        14,  # 15: tail2 -> tail1
        15,  # 16: tail3 -> tail2
        16,  # 17: tail4 -> tail3
        17,  # 18: tail5 -> tail4
        7,   # 19: left_shoulder -> neck
        19,  # 20: left_midside -> left_shoulder
        20,  # 21: left_hip -> left_midside
        7,   # 22: right_shoulder -> neck
        22,  # 23: right_midside -> right_shoulder
        23,  # 24: right_hip -> right_midside
        18,  # 25: tail_end -> tail5
        -1,  # 26: head_midpoint (root)
    ],
    edges=[
        # Head
        (26, 0), (26, 1), (26, 2), (26, 5), (26, 6), (26, 7),
        (1, 3), (2, 4),
        # Spine
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
        # Tail
        (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 25),
        # Left side
        (7, 19), (19, 20), (20, 21),
        # Right side
        (7, 22), (22, 23), (23, 24),
    ],
    symmetric_pairs=[
        (1, 2), (3, 4), (5, 6),
        (19, 22), (20, 23), (21, 24),
    ],
    num_channels=2,
    coordinate_system="xy",
    body_parts={
        "head": [0, 1, 2, 3, 4, 5, 6, 26],
        "spine": [7, 8, 9, 10, 11, 12],
        "tail": [13, 14, 15, 16, 17, 18, 25],
        "left_side": [19, 20, 21],
        "right_side": [22, 23, 24],
    },
    center_joint=9,
)


DLC_QUADRUPED_SKELETON = SkeletonDefinition(
    name="dlc_quadruped",
    num_joints=39,
    joint_names=[
        "nose",                # 0
        "upper_jaw",           # 1
        "lower_jaw",           # 2
        "mouth_end_right",     # 3
        "mouth_end_left",      # 4
        "right_eye",           # 5
        "right_earbase",       # 6
        "right_earend",        # 7
        "right_antler_base",   # 8
        "right_antler_end",    # 9
        "left_eye",            # 10
        "left_earbase",        # 11
        "left_earend",         # 12
        "left_antler_base",    # 13
        "left_antler_end",     # 14
        "neck_base",           # 15
        "neck_end",            # 16
        "throat_base",         # 17
        "throat_end",          # 18
        "back_base",           # 19
        "back_end",            # 20
        "back_middle",         # 21
        "tail_base",           # 22
        "tail_end",            # 23
        "tail_middle",         # 24
        "left_front_hoof",     # 25
        "left_front_knee",     # 26
        "left_front_paw",      # 27
        "left_front_elbow",    # 28
        "right_front_hoof",    # 29
        "right_front_knee",    # 30
        "right_front_paw",     # 31
        "right_front_elbow",   # 32
        "left_back_hoof",      # 33
        "left_back_knee",      # 34
        "left_back_paw",       # 35
        "left_back_elbow",     # 36
        "right_back_hoof",     # 37
        "right_back_knee",     # 38
    ],
    joint_parents=[
        15,  # 0: nose -> neck_base
        0,   # 1: upper_jaw -> nose
        0,   # 2: lower_jaw -> nose
        0,   # 3: mouth_end_right -> nose
        0,   # 4: mouth_end_left -> nose
        0,   # 5: right_eye -> nose
        5,   # 6: right_earbase -> right_eye
        6,   # 7: right_earend -> right_earbase
        6,   # 8: right_antler_base -> right_earbase
        8,   # 9: right_antler_end -> right_antler_base
        0,   # 10: left_eye -> nose
        10,  # 11: left_earbase -> left_eye
        11,  # 12: left_earend -> left_earbase
        11,  # 13: left_antler_base -> left_earbase
        13,  # 14: left_antler_end -> left_antler_base
        -1,  # 15: neck_base (root)
        15,  # 16: neck_end -> neck_base
        16,  # 17: throat_base -> neck_end
        17,  # 18: throat_end -> throat_base
        15,  # 19: back_base -> neck_base
        19,  # 20: back_end -> back_base
        19,  # 21: back_middle -> back_base
        20,  # 22: tail_base -> back_end
        22,  # 23: tail_end -> tail_base
        22,  # 24: tail_middle -> tail_base
        28,  # 25: left_front_hoof -> left_front_elbow
        28,  # 26: left_front_knee -> left_front_elbow
        26,  # 27: left_front_paw -> left_front_knee
        15,  # 28: left_front_elbow -> neck_base
        32,  # 29: right_front_hoof -> right_front_elbow
        32,  # 30: right_front_knee -> right_front_elbow
        30,  # 31: right_front_paw -> right_front_knee
        15,  # 32: right_front_elbow -> neck_base
        36,  # 33: left_back_hoof -> left_back_elbow
        36,  # 34: left_back_knee -> left_back_elbow
        34,  # 35: left_back_paw -> left_back_knee
        20,  # 36: left_back_elbow -> back_end
        20,  # 37: right_back_hoof -> back_end
        20,  # 38: right_back_knee -> back_end
    ],
    edges=[
        # Head
        (15, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (0, 5), (5, 6), (6, 7), (6, 8), (8, 9),
        (0, 10), (10, 11), (11, 12), (11, 13), (13, 14),
        # Neck/throat
        (15, 16), (16, 17), (17, 18),
        # Back
        (15, 19), (19, 21), (19, 20),
        # Tail
        (20, 22), (22, 23), (22, 24),
        # Front legs
        (15, 28), (28, 26), (26, 27), (28, 25),
        (15, 32), (32, 30), (30, 31), (32, 29),
        # Back legs
        (20, 36), (36, 34), (34, 35), (36, 33),
        (20, 38), (38, 37),
    ],
    symmetric_pairs=[
        (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),  # head
        (3, 4),   # mouth
        (25, 29), (26, 30), (27, 31), (28, 32),  # front legs
        (33, 37), (34, 38),  # back legs
    ],
    num_channels=2,
    coordinate_system="xy",
    body_parts={
        "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "neck": [15, 16, 17, 18],
        "back": [19, 20, 21],
        "tail": [22, 23, 24],
        "left_front_leg": [25, 26, 27, 28],
        "right_front_leg": [29, 30, 31, 32],
        "left_back_leg": [33, 34, 35, 36],
        "right_back_leg": [37, 38],
    },
    center_joint=15,
)


# =============================================================================
# Skeleton Registry
# =============================================================================

SKELETON_REGISTRY: dict[str, SkeletonDefinition] = {
    # Human
    "ntu": NTU_SKELETON,
    "ntu_rgbd": NTU_SKELETON,
    "ntu60": NTU_SKELETON,
    "ntu120": NTU_SKELETON,
    "ntu_rgb_d": NTU_SKELETON,
    "ucla": UCLA_SKELETON,
    "nw_ucla": UCLA_SKELETON,
    "nwucla": UCLA_SKELETON,
    "coco": COCO_SKELETON,
    "coco17": COCO_SKELETON,
    # Animal
    "mars": MARS_MOUSE_SKELETON,
    "mars_mouse": MARS_MOUSE_SKELETON,
    "calms21": CALMS21_MOUSE_SKELETON,
    "calms21_mouse": CALMS21_MOUSE_SKELETON,
    # 3D Rodent
    "rat7m": RAT7M_SKELETON,
    "rat7m_20": RAT7M_SKELETON,
    "subtle": SUBTLE_MOUSE_SKELETON,
    "subtle_mouse": SUBTLE_MOUSE_SKELETON,
    "mabe22": MABE22_MOUSE_SKELETON,
    "mabe22_mouse": MABE22_MOUSE_SKELETON,
    "mabe": MABE22_MOUSE_SKELETON,
    "shank3ko": SHANK3KO_MOUSE_SKELETON,
    "shank3ko_mouse": SHANK3KO_MOUSE_SKELETON,
    # DLC
    "dlc_topviewmouse": DLC_TOPVIEWMOUSE_SKELETON,
    "dlc_topviewmouse27": DLC_TOPVIEWMOUSE_SKELETON,
    "topviewmouse": DLC_TOPVIEWMOUSE_SKELETON,
    "dlc_quadruped": DLC_QUADRUPED_SKELETON,
    "dlc_quadruped39": DLC_QUADRUPED_SKELETON,
    "quadruped": DLC_QUADRUPED_SKELETON,
}


def register_skeleton(name: str, skeleton: SkeletonDefinition) -> None:
    """Register a new skeleton definition at runtime."""
    SKELETON_REGISTRY[name.lower()] = skeleton


def get_skeleton(name: str) -> SkeletonDefinition:
    """Get skeleton definition by name.

    Args:
        name: Skeleton identifier (e.g., 'ntu', 'mars', 'dlc_topviewmouse')

    Raises:
        ValueError: If skeleton name not found
    """
    key = name.lower()
    if key not in SKELETON_REGISTRY:
        unique = sorted({s.name for s in SKELETON_REGISTRY.values()})
        raise ValueError(f"Unknown skeleton: '{name}'. Available: {unique}")
    return SKELETON_REGISTRY[key]


def list_skeletons() -> list[str]:
    """List unique skeleton names (not aliases)."""
    seen: set[str] = set()
    names: list[str] = []
    for name, skel in SKELETON_REGISTRY.items():
        if skel.name not in seen:
            names.append(name)
            seen.add(skel.name)
    return sorted(names)


def get_skeleton_info(name: str) -> dict[str, Any]:
    """Get summary info about a skeleton."""
    s = get_skeleton(name)
    return {
        "name": s.name,
        "num_joints": s.num_joints,
        "num_edges": len(s.edges),
        "num_channels": s.num_channels,
        "coordinate_system": s.coordinate_system,
        "body_parts": list(s.body_parts.keys()),
        "symmetric_pairs": len(s.symmetric_pairs),
        "joint_names": s.joint_names,
        "num_persons": s.num_persons,
    }


def load_skeleton_from_yaml(yaml_path: str | Path) -> SkeletonDefinition:
    """Load a skeleton definition from a YAML config file.

    Expected YAML format:
        name: my_skeleton
        num_joints: 10
        joint_names: [a, b, c, ...]
        joint_parents: [-1, 0, 1, ...]
        edges: [[0, 1], [1, 2], ...]
        # optional fields...
    """
    import yaml

    path = Path(yaml_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    edges = [tuple(e) for e in cfg["edges"]]
    sym = [tuple(p) for p in cfg.get("symmetric_pairs", [])]

    body_parts = {}
    for k, v in cfg.get("body_parts", {}).items():
        body_parts[k] = list(v)

    return SkeletonDefinition(
        name=cfg["name"],
        num_joints=cfg["num_joints"],
        joint_names=cfg["joint_names"],
        joint_parents=cfg["joint_parents"],
        edges=edges,
        symmetric_pairs=sym,
        num_channels=cfg.get("num_channels", 3),
        coordinate_system=cfg.get("coordinate_system", "xyz"),
        body_parts=body_parts,
        center_joint=cfg.get("center_joint", 0),
        num_persons=cfg.get("num_persons", 1),
    )
