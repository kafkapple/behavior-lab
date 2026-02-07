"""Core module — numpy-only, zero heavy dependencies.

Public API:
    get_skeleton, list_skeletons, register_skeleton, load_skeleton_from_yaml
    SkeletonDefinition, get_skeleton_info
    Graph, get_graph
    sequence_to_graph, graph_to_sequence
    BehaviorSequence, ClassificationResult, ModelMetrics
"""
from .skeleton import (
    SkeletonDefinition,
    get_skeleton,
    get_skeleton_info,
    list_skeletons,
    load_skeleton_from_yaml,
    register_skeleton,
)
from .graph import Graph, get_graph
from .tensor_format import graph_to_sequence, sequence_to_graph
from .types import BehaviorSequence, ClassificationResult, ModelMetrics

__all__ = [
    "SkeletonDefinition",
    "get_skeleton",
    "get_skeleton_info",
    "list_skeletons",
    "register_skeleton",
    "load_skeleton_from_yaml",
    "Graph",
    "get_graph",
    "sequence_to_graph",
    "graph_to_sequence",
    "BehaviorSequence",
    "ClassificationResult",
    "ModelMetrics",
]
