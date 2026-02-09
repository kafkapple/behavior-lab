"""Unified feature extraction: kinematic + morphometric."""
import numpy as np
from typing import Dict, List, Optional
from scipy.signal import savgol_filter


def compute_velocity(keypoints: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """Compute per-frame speed from centroid displacement.
    
    Args:
        keypoints: (T, K, D) where D >= 2
        smooth_window: Savitzky-Golay filter window
        
    Returns:
        velocity: (T,)
    """
    centroid = keypoints[:, :, :2].mean(axis=1)  # (T, 2)
    disp = np.diff(centroid, axis=0, prepend=centroid[:1])
    speed = np.linalg.norm(disp, axis=1)

    if len(speed) > smooth_window and smooth_window > 2:
        w = min(smooth_window, len(speed) // 2 * 2 - 1)
        if w >= 3:
            speed = savgol_filter(speed, w, 2)

    return speed


def compute_acceleration(velocity: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """Compute acceleration from velocity."""
    accel = np.diff(velocity, prepend=velocity[0])
    if len(accel) > smooth_window and smooth_window > 2:
        w = min(smooth_window, len(accel) // 2 * 2 - 1)
        if w >= 3:
            accel = savgol_filter(accel, w, 2)
    return accel


def compute_body_spread(keypoints: np.ndarray) -> np.ndarray:
    """Max spread of keypoints per frame (proxy for posture).
    
    Args:
        keypoints: (T, K, D)
    Returns:
        spread: (T,)
    """
    xy = keypoints[:, :, :2]
    return np.max(np.std(xy, axis=1), axis=1)


def compute_spatial_variance(keypoints: np.ndarray) -> np.ndarray:
    """Variance of keypoint positions per frame (contraction/expansion).
    
    Args:
        keypoints: (T, K, D)
    Returns:
        variance: (T,)
    """
    xy = keypoints[:, :, :2]
    return np.var(xy, axis=(1, 2))


def estimate_body_size(
    keypoints: np.ndarray,
    joint_names: Optional[List[str]] = None,
    head_idx: int = 0,
    tail_idx: int = -1,
) -> float:
    """Estimate body size as median head-tail distance.
    
    Args:
        keypoints: (T, K, D)
        joint_names: optional joint name list to find nose/tail
        head_idx: index of head joint
        tail_idx: index of tail joint
    """
    if joint_names:
        for name in ['nose', 'head']:
            if name in joint_names:
                head_idx = joint_names.index(name)
                break
        for name in ['tail_base', 'tail_end']:
            if name in joint_names:
                tail_idx = joint_names.index(name)
                break

    head = keypoints[:, head_idx, :2]
    tail = keypoints[:, tail_idx, :2]
    distances = np.linalg.norm(head - tail, axis=1)
    return float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0


def extract_features(
    keypoints: np.ndarray,
    smooth_window: int = 5,
    fps: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Extract all features from keypoint sequence.
    
    Args:
        keypoints: (T, K, D) canonical format
        smooth_window: smoothing window
        fps: frames per second
        
    Returns:
        dict with 'speed', 'acceleration', 'body_spread', 'spatial_variance',
        'feature_matrix' (T, 4)
    """
    speed = compute_velocity(keypoints, smooth_window)
    accel = compute_acceleration(speed, smooth_window)
    spread = compute_body_spread(keypoints)
    variance = compute_spatial_variance(keypoints)

    return {
        'speed': speed,
        'acceleration': accel,
        'body_spread': spread,
        'spatial_variance': variance,
        'feature_matrix': np.column_stack([speed, accel, spread, variance]),
        'total_distance': float(np.sum(speed)),
        'mean_speed': float(np.mean(speed)),
    }


class FeatureExtractor:
    """Stateful feature extractor with body-size normalization.
    
    Args:
        fps: frames per second
        smooth_window: smoothing window
        normalize_body_size: normalize velocities by body size
    """

    def __init__(self, fps: float = 30.0, smooth_window: int = 5,
                 normalize_body_size: bool = False):
        self.fps = fps
        self.smooth_window = smooth_window
        self.normalize_body_size = normalize_body_size

    def __call__(self, keypoints: np.ndarray,
                 joint_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Extract features from (T, K, D) keypoints."""
        features = extract_features(keypoints, self.smooth_window, self.fps)

        if self.normalize_body_size:
            body_size = estimate_body_size(keypoints, joint_names)
            features['speed'] = features['speed'] / body_size * self.fps
            features['acceleration'] = features['acceleration'] / body_size * self.fps
            features['body_size'] = body_size
            features['feature_matrix'] = np.column_stack([
                features['speed'], features['acceleration'],
                features['body_spread'], features['spatial_variance']
            ])

        return features
