"""Visualization utilities for behavior analysis."""
from .agreement import label_agreement, match_clusters, plot_label_agreement, stretch_labels
from .analysis import (
    plot_behavior_dendrogram,
    plot_bout_duration,
    plot_hierarchical_embeddings,
    plot_multiscale_ethogram,
    plot_temporal_raster,
    plot_transition_matrix,
)
from .cluster_map import (
    cluster_sizes,
    plot_cluster_map,
    plot_cluster_sizes,
    plot_method_maps,
    plot_subtle_cluster_map,
    pose_embedding,
    rank_colors,
    transition_matrix,
)
from .colors import (
    BODY_PART_COLORS,
    PERSON_COLORS,
    get_joint_colors,
    get_joint_full_names,
    get_joint_labels,
    get_limb_colors,
    get_person_colors,
)
from .comparison import render_cluster_gallery, render_comparison_report
from .dynamics import common_rate, plot_dynamics
from .embedding import plot_embedding, plot_embedding_3d
from .html_report import fig_to_base64, generate_pipeline_report
from .keypoint_schema import plot_keypoint_schema
from .player import PLAYER_JS, player_block, player_data
from .skeleton import (
    animate_skeleton,
    plot_skeleton,
    plot_skeleton_comparison,
    strip_zero_frames,
    strip_zero_persons,
)
from .video_overlay import (
    overlay_keypoints_on_frame_array,
    overlay_keypoints_on_video,
    render_skeleton_on_frame,
)

__all__ = [
    "common_rate",
    "plot_dynamics",
    "rank_colors",
    "label_agreement",
    "plot_cluster_map",
    "plot_label_agreement",
    "plot_subtle_cluster_map",
    "stretch_labels",
    "transition_matrix",
    "PLAYER_JS",
    "cluster_sizes",
    "match_clusters",
    "player_block",
    "player_data",
    "plot_cluster_sizes",
    "plot_keypoint_schema",
    "plot_method_maps",
    "pose_embedding",
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
    "render_comparison_report",
    "render_cluster_gallery",
    "render_skeleton_on_frame",
    "overlay_keypoints_on_video",
    "overlay_keypoints_on_frame_array",
]
