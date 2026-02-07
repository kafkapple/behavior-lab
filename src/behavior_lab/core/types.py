"""
Core data types and protocols for behavior-lab.

Defines shared data structures used across the pipeline:
- BehaviorSequence: canonical skeleton sequence container
- ClassificationResult: model prediction output
- Protocols for model and feeder interfaces
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class BehaviorSequence:
    """Canonical skeleton sequence container.

    Stores keypoint data in (T, K, D) format with optional labels and metadata.

    Attributes:
        keypoints: Shape (T, K, D) — frames x joints x channels
        labels: Optional per-frame labels, shape (T,)
        skeleton_name: Name of the skeleton definition used
        sample_id: Optional identifier for this sequence
        fps: Frames per second (for temporal analysis)
        metadata: Arbitrary key-value metadata
    """
    keypoints: np.ndarray  # (T, K, D)
    labels: np.ndarray | None = None  # (T,)
    skeleton_name: str = ""
    sample_id: str = ""
    fps: float = 30.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.keypoints.shape[0]

    @property
    def num_joints(self) -> int:
        return self.keypoints.shape[1]

    @property
    def num_channels(self) -> int:
        return self.keypoints.shape[2]

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    def validate(self) -> None:
        """Check internal consistency."""
        assert self.keypoints.ndim == 3, (
            f"Expected 3D keypoints (T,K,D), got shape {self.keypoints.shape}"
        )
        if self.labels is not None:
            assert self.labels.shape[0] == self.num_frames, (
                f"Label length {self.labels.shape[0]} != frames {self.num_frames}"
            )


@dataclass
class ClassificationResult:
    """Output of an action/behavior classifier.

    Attributes:
        predictions: Per-frame class predictions, shape (T,)
        probabilities: Per-frame class probabilities, shape (T, C)
        class_names: Ordered class label names
        model_name: Name of the model that produced this result
    """
    predictions: np.ndarray  # (T,)
    probabilities: np.ndarray | None = None  # (T, num_classes)
    class_names: list[str] = field(default_factory=list)
    model_name: str = ""

    @property
    def num_frames(self) -> int:
        return self.predictions.shape[0]

    @property
    def num_classes(self) -> int:
        if self.probabilities is not None:
            return self.probabilities.shape[-1]
        return len(self.class_names) if self.class_names else 0


@dataclass
class ModelMetrics:
    """Evaluation metrics for a classifier.

    Attributes:
        accuracy: Overall accuracy
        f1_macro: Macro-averaged F1 score
        per_class_accuracy: Per-class accuracy dict
        confusion_matrix: (C, C) confusion matrix
    """
    accuracy: float = 0.0
    f1_macro: float = 0.0
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    confusion_matrix: np.ndarray | None = None


# =============================================================================
# Protocols — structural typing for model/feeder interfaces
# =============================================================================

@runtime_checkable
class ActionClassifier(Protocol):
    """Protocol for sequence-based action classifiers."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict: ...

    def predict(self, X: np.ndarray) -> ClassificationResult: ...

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelMetrics: ...


@runtime_checkable
class PoseEstimator(Protocol):
    """Protocol for pose estimation backends."""

    def predict_video(
        self, video_path: str, max_frames: int | None = None
    ) -> BehaviorSequence: ...
