"""Visualization utilities for behavior analysis."""
from .embedding import plot_embedding, plot_embedding_3d
from .skeleton import plot_skeleton, animate_skeleton
from .analysis import plot_transition_matrix, plot_bout_duration, plot_temporal_raster

__all__ = [
    "plot_embedding",
    "plot_embedding_3d",
    "plot_skeleton",
    "animate_skeleton",
    "plot_transition_matrix",
    "plot_bout_duration",
    "plot_temporal_raster",
]
