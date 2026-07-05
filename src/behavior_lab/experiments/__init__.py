"""Reusable experiment runners for notebook-first behavior analysis."""

from .discovery import DiscoveryRun, compare_discovery_methods, extract_feature_matrix
from .pipeline import DATASET_SPECS, DatasetSpec, run_comparison

__all__ = ["DiscoveryRun", "compare_discovery_methods", "extract_feature_matrix",
           "DatasetSpec", "DATASET_SPECS", "run_comparison"]
