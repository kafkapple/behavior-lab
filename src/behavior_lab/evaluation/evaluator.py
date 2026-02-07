"""Evaluation metrics and evaluator for behavior recognition."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    normalized_mutual_info_score, adjusted_rand_score,
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, v_measure_score,
    homogeneity_score, completeness_score,
)
from scipy.optimize import linear_sum_assignment


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation results."""
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_per_class: Dict[str, float] = field(default_factory=dict)
    confusion: Optional[np.ndarray] = None
    num_samples: int = 0


@dataclass
class ClusterMetrics:
    """Container for unsupervised/SSL clustering evaluation."""
    nmi: float = 0.0
    ari: float = 0.0
    silhouette: float = 0.0
    calinski_harabasz: float = 0.0
    davies_bouldin: float = 0.0
    v_measure: float = 0.0
    homogeneity: float = 0.0
    completeness: float = 0.0
    hungarian_accuracy: float = 0.0
    num_clusters: int = 0


@dataclass
class BehaviorMetrics:
    """Behavior-specific metrics for temporal sequence analysis.

    Attributes:
        bout_durations: Per-class mean bout duration in seconds
        transition_matrix: (C, C) transition probability matrix
        temporal_consistency: Fraction of same-label consecutive pairs
        num_bouts: Total number of behavioral bouts
        entropy_rate: Entropy rate of the label sequence
    """
    bout_durations: Dict[int, float] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = None
    temporal_consistency: float = 0.0
    num_bouts: int = 0
    entropy_rate: float = 0.0


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """Compute supervised classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class F1
    classes = class_names or [str(i) for i in sorted(set(y_true))]
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_dict = {name: float(f) for name, f in zip(classes, f1_per)}

    return ClassificationMetrics(
        accuracy=acc, f1_macro=f1_mac, f1_per_class=f1_dict,
        confusion=cm, num_samples=len(y_true)
    )


def compute_cluster_metrics(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
) -> ClusterMetrics:
    """Compute clustering quality metrics.

    Args:
        features: (N, D) feature vectors
        cluster_labels: (N,) predicted cluster assignments
        true_labels: (N,) ground truth labels (optional)
    """
    n_clusters = len(set(cluster_labels))
    metrics = ClusterMetrics(num_clusters=n_clusters)

    # Internal metrics (no labels needed)
    if n_clusters > 1 and n_clusters < len(features):
        metrics.silhouette = silhouette_score(features, cluster_labels)
        metrics.calinski_harabasz = calinski_harabasz_score(features, cluster_labels)
        metrics.davies_bouldin = davies_bouldin_score(features, cluster_labels)

    # External metrics (need labels)
    if true_labels is not None:
        metrics.nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        metrics.ari = adjusted_rand_score(true_labels, cluster_labels)
        metrics.v_measure = v_measure_score(true_labels, cluster_labels)
        metrics.homogeneity = homogeneity_score(true_labels, cluster_labels)
        metrics.completeness = completeness_score(true_labels, cluster_labels)
        metrics.hungarian_accuracy = _hungarian_accuracy(true_labels, cluster_labels)

    return metrics


def _hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optimal 1:1 cluster-to-class assignment via Hungarian algorithm."""
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    # Build cost matrix
    cost = np.zeros((len(clusters), len(classes)))
    for i, c in enumerate(clusters):
        mask = y_pred == c
        for j, cl in enumerate(classes):
            cost[i, j] = -np.sum(y_true[mask] == cl)

    row_idx, col_idx = linear_sum_assignment(cost)

    # Map clusters to classes
    mapping = {clusters[r]: classes[c] for r, c in zip(row_idx, col_idx)}
    mapped = np.array([mapping.get(p, -1) for p in y_pred])
    return accuracy_score(y_true, mapped)


def compute_behavior_metrics(
    labels: np.ndarray,
    fps: float = 30.0,
) -> BehaviorMetrics:
    """Compute behavior-specific temporal metrics from label sequences.

    Args:
        labels: (T,) per-frame behavior labels
        fps: Frames per second for duration calculation

    Returns:
        BehaviorMetrics with bout durations, transition matrix, etc.
    """
    T = len(labels)
    if T == 0:
        return BehaviorMetrics()

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # Bout analysis: consecutive runs of same label
    bout_durations: Dict[int, list] = {l: [] for l in unique_labels}
    current_label = labels[0]
    current_length = 1

    for t in range(1, T):
        if labels[t] == current_label:
            current_length += 1
        else:
            bout_durations[current_label].append(current_length / fps)
            current_label = labels[t]
            current_length = 1
    bout_durations[current_label].append(current_length / fps)

    mean_durations = {
        l: float(np.mean(durs)) if durs else 0.0
        for l, durs in bout_durations.items()
    }
    num_bouts = sum(len(durs) for durs in bout_durations.values())

    # Transition matrix
    trans = np.zeros((n_classes, n_classes))
    for t in range(T - 1):
        i = label_to_idx[labels[t]]
        j = label_to_idx[labels[t + 1]]
        trans[i, j] += 1

    # Normalize rows to probabilities
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans / row_sums

    # Temporal consistency: fraction of same-label consecutive pairs
    same_pairs = np.sum(labels[:-1] == labels[1:])
    temporal_consistency = float(same_pairs / (T - 1)) if T > 1 else 1.0

    # Entropy rate
    entropy_rate = 0.0
    for i in range(n_classes):
        pi = trans[i].sum() / trans.sum() if trans.sum() > 0 else 0
        for j in range(n_classes):
            if trans_prob[i, j] > 0:
                entropy_rate -= pi * trans_prob[i, j] * np.log2(trans_prob[i, j])

    return BehaviorMetrics(
        bout_durations=mean_durations,
        transition_matrix=trans_prob,
        temporal_consistency=temporal_consistency,
        num_bouts=num_bouts,
        entropy_rate=float(entropy_rate),
    )


def linear_probe(
    train_features: np.ndarray, train_labels: np.ndarray,
    test_features: np.ndarray, test_labels: np.ndarray,
) -> ClassificationMetrics:
    """Train linear classifier on frozen features (downstream probing)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)

    return compute_classification_metrics(test_labels, y_pred)


class Evaluator:
    """Unified evaluator supporting supervised and unsupervised paradigms.

    Usage:
        evaluator = Evaluator(class_names=['other', 'attack', 'mount', 'investigation'])
        metrics = evaluator.evaluate_supervised(y_true, y_pred)
        metrics = evaluator.evaluate_clusters(features, cluster_labels, y_true)
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names

    def evaluate_supervised(self, y_true, y_pred) -> ClassificationMetrics:
        """Evaluate supervised predictions."""
        return compute_classification_metrics(
            np.asarray(y_true), np.asarray(y_pred), self.class_names
        )

    def evaluate_clusters(self, features, cluster_labels, true_labels=None) -> ClusterMetrics:
        """Evaluate clustering quality."""
        return compute_cluster_metrics(
            np.asarray(features), np.asarray(cluster_labels),
            np.asarray(true_labels) if true_labels is not None else None
        )

    def evaluate_behavior(self, labels: np.ndarray, fps: float = 30.0) -> BehaviorMetrics:
        """Evaluate temporal behavior metrics."""
        return compute_behavior_metrics(np.asarray(labels), fps)

    def evaluate_linear_probe(self, train_feat, train_labels, test_feat, test_labels) -> ClassificationMetrics:
        """Evaluate frozen features via linear probing."""
        return linear_probe(
            np.asarray(train_feat), np.asarray(train_labels),
            np.asarray(test_feat), np.asarray(test_labels)
        )

    def print_report(self, metrics) -> str:
        """Format metrics as a readable report string."""
        lines = []
        if isinstance(metrics, ClassificationMetrics):
            lines.append(f"Accuracy: {metrics.accuracy:.4f}")
            lines.append(f"F1 (macro): {metrics.f1_macro:.4f}")
            for name, f1 in metrics.f1_per_class.items():
                lines.append(f"  {name}: F1={f1:.4f}")
        elif isinstance(metrics, ClusterMetrics):
            lines.append(f"NMI: {metrics.nmi:.4f}")
            lines.append(f"ARI: {metrics.ari:.4f}")
            lines.append(f"Silhouette: {metrics.silhouette:.4f}")
            lines.append(f"Calinski-Harabasz: {metrics.calinski_harabasz:.1f}")
            lines.append(f"Davies-Bouldin: {metrics.davies_bouldin:.4f}")
            lines.append(f"V-Measure: {metrics.v_measure:.4f}")
            lines.append(f"Hungarian Accuracy: {metrics.hungarian_accuracy:.4f}")
        elif isinstance(metrics, BehaviorMetrics):
            lines.append(f"Num Bouts: {metrics.num_bouts}")
            lines.append(f"Temporal Consistency: {metrics.temporal_consistency:.4f}")
            lines.append(f"Entropy Rate: {metrics.entropy_rate:.4f}")
            for label, dur in metrics.bout_durations.items():
                lines.append(f"  Class {label}: mean bout = {dur:.2f}s")
        report = '\n'.join(lines)
        print(report)
        return report
