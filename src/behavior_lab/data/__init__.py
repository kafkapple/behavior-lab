"""Data loading, preprocessing, and feature extraction."""
from .preprocessing.augmentation import SkeletonAugmentor, random_rot, valid_crop_resize
from .features.features import extract_features, FeatureExtractor

try:
    from .feeders.skeleton_feeder import SkeletonFeeder, get_feeder
except ImportError:
    SkeletonFeeder = None

    def get_feeder(*args, **kwargs):
        raise ImportError("Install behavior-lab[torch] to use SkeletonFeeder") from None


__all__ = [
    "SkeletonFeeder",
    "get_feeder",
    "SkeletonAugmentor",
    "random_rot",
    "valid_crop_resize",
    "extract_features",
    "FeatureExtractor",
]
