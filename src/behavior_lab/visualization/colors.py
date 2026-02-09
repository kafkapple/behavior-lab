"""Semantic body-part color palette and mapping utilities.

Maps skeleton body_parts to visually distinct colors for rendering.
Supports multi-person distinction via PERSON_COLORS.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from behavior_lab.core.skeleton import SkeletonDefinition

# Semantic body-part color palette
BODY_PART_COLORS: dict[str, str] = {
    # Head / Face
    "head": "#E74C3C",
    "face": "#E74C3C",
    # Torso / Spine
    "torso": "#3498DB",
    "spine": "#3498DB",
    "body": "#3498DB",
    "neck": "#2980B9",
    "back": "#2980B9",
    # Arms (human)
    "left_arm": "#2ECC71",
    "right_arm": "#F39C12",
    # Legs (human)
    "left_leg": "#9B59B6",
    "right_leg": "#E67E22",
    # Animal sides
    "left_side": "#2ECC71",
    "right_side": "#F39C12",
    "tail": "#D4AC0D",
    # Quadruped legs
    "left_front_leg": "#2ECC71",
    "right_front_leg": "#F39C12",
    "left_back_leg": "#9B59B6",
    "right_back_leg": "#E67E22",
    "front_left": "#2ECC71",
    "front_right": "#F39C12",
    "hind_left": "#9B59B6",
    "hind_right": "#E67E22",
    # Paws (MABe22-style naming)
    "front_paws": "#1ABC9C",
    "hind_paws": "#8E44AD",
}

_FALLBACK_COLOR = "#7F8C8D"

# Multi-person palette (maximally distinct)
PERSON_COLORS: list[str] = [
    "#1ABC9C",  # Teal
    "#E74C3C",  # Red
    "#3498DB",  # Blue
    "#F39C12",  # Orange
    "#9B59B6",  # Purple
    "#2ECC71",  # Green
]


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to BGR tuple for OpenCV.

    '#E74C3C' -> (60, 76, 231)
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _joint_to_part(skeleton: SkeletonDefinition) -> dict[int, str]:
    """Build reverse mapping: joint_index -> body_part name."""
    mapping: dict[int, str] = {}
    for part_name, indices in skeleton.body_parts.items():
        for idx in indices:
            if idx not in mapping:
                mapping[idx] = part_name
    return mapping


def get_joint_colors(skeleton: SkeletonDefinition) -> list[str]:
    """Return per-joint color list based on body_parts.

    Falls back to skeleton.joint_colors if set, otherwise derives
    from BODY_PART_COLORS using the skeleton's body_parts mapping.
    """
    if skeleton.joint_colors is not None:
        return skeleton.joint_colors

    if not skeleton.body_parts:
        return [_FALLBACK_COLOR] * skeleton.num_joints

    j2p = _joint_to_part(skeleton)
    return [
        BODY_PART_COLORS.get(j2p.get(i, ""), _FALLBACK_COLOR)
        for i in range(skeleton.num_joints)
    ]


def get_limb_colors(skeleton: SkeletonDefinition) -> list[str]:
    """Return per-edge color list.

    Edge color = body_part color if both endpoints share the same part,
    otherwise the color of the first endpoint's part.
    """
    if skeleton.limb_colors is not None:
        return skeleton.limb_colors

    if not skeleton.body_parts:
        return [_FALLBACK_COLOR] * len(skeleton.edges)

    j2p = _joint_to_part(skeleton)
    colors: list[str] = []
    for i, j in skeleton.edges:
        part_i = j2p.get(i, "")
        part_j = j2p.get(j, "")
        if part_i == part_j:
            colors.append(BODY_PART_COLORS.get(part_i, _FALLBACK_COLOR))
        else:
            colors.append(BODY_PART_COLORS.get(part_i, _FALLBACK_COLOR))
    return colors


def get_person_colors(n_persons: int) -> list[str]:
    """Return a list of distinct colors for multi-person rendering."""
    return [PERSON_COLORS[i % len(PERSON_COLORS)] for i in range(n_persons)]


def get_joint_full_names(skeleton: SkeletonDefinition) -> list[str]:
    """Return full joint names formatted for display.

    'left_shoulder' -> 'Left Shoulder', 'tail_base' -> 'Tail Base'
    """
    return [name.replace("_", " ").title() for name in skeleton.joint_names]


def get_joint_labels(skeleton: SkeletonDefinition) -> list[str]:
    """Return abbreviated joint labels for overlay display.

    Generates short labels from joint_names:
    'left_shoulder' -> 'LSh', 'nose' -> 'Nos', 'tail_base' -> 'TBa'
    """
    labels: list[str] = []
    for name in skeleton.joint_names:
        parts = name.split("_")
        if len(parts) == 1:
            labels.append(name[:3].capitalize())
        elif len(parts) == 2:
            labels.append(parts[0][0].upper() + parts[1][:2].capitalize())
        else:
            labels.append("".join(p[0].upper() for p in parts[:3]))
    return labels
