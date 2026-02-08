"""Visualization utilities for behavior analysis."""
from .embedding import plot_embedding, plot_embedding_3d
from .skeleton import plot_skeleton, animate_skeleton, plot_skeleton_comparison
from .analysis import plot_transition_matrix, plot_bout_duration, plot_temporal_raster
from .colors import (
    get_joint_colors,
    get_limb_colors,
    get_person_colors,
    BODY_PART_COLORS,
    PERSON_COLORS,
)
from .html_report import generate_pipeline_report, fig_to_base64

__all__ = [
    "plot_embedding",
    "plot_embedding_3d",
    "plot_skeleton",
    "animate_skeleton",
    "plot_skeleton_comparison",
    "plot_transition_matrix",
    "plot_bout_duration",
    "plot_temporal_raster",
    "get_joint_colors",
    "get_limb_colors",
    "get_person_colors",
    "BODY_PART_COLORS",
    "PERSON_COLORS",
    "generate_pipeline_report",
    "fig_to_base64",
]
