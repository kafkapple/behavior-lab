"""Visualization utilities for behavior analysis."""
from .embedding import plot_embedding, plot_embedding_3d
from .skeleton import (
    plot_skeleton, animate_skeleton, plot_skeleton_comparison,
    strip_zero_frames, strip_zero_persons,
)
from .analysis import (
    plot_transition_matrix,
    plot_bout_duration,
    plot_temporal_raster,
    plot_multiscale_ethogram,
    plot_hierarchical_embeddings,
    plot_behavior_dendrogram,
)
from .colors import (
    get_joint_colors,
    get_limb_colors,
    get_person_colors,
    get_joint_labels,
    get_joint_full_names,
    BODY_PART_COLORS,
    PERSON_COLORS,
)
from .html_report import generate_pipeline_report, fig_to_base64
from .video_overlay import (
    render_skeleton_on_frame,
    overlay_keypoints_on_video,
    overlay_keypoints_on_frame_array,
)

__all__ = [
    "plot_embedding",
    "plot_embedding_3d",
    "plot_skeleton",
    "animate_skeleton",
    "plot_skeleton_comparison",
    "strip_zero_frames",
    "strip_zero_persons",
    "plot_transition_matrix",
    "plot_bout_duration",
    "plot_temporal_raster",
    "plot_multiscale_ethogram",
    "plot_hierarchical_embeddings",
    "plot_behavior_dendrogram",
    "get_joint_colors",
    "get_limb_colors",
    "get_person_colors",
    "get_joint_labels",
    "get_joint_full_names",
    "BODY_PART_COLORS",
    "PERSON_COLORS",
    "generate_pipeline_report",
    "fig_to_base64",
    "render_skeleton_on_frame",
    "overlay_keypoints_on_video",
    "overlay_keypoints_on_frame_array",
]
