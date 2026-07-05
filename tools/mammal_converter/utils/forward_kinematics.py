"""
Forward Kinematics utilities for skeleton-based animation.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def forward_kinematics(
    local_transforms: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """
    Compute global transforms from local transforms using forward kinematics.

    Args:
        local_transforms: (num_joints, 4, 4) local transformation matrices
        parents: (num_joints,) parent joint indices (-1 for root)

    Returns:
        global_transforms: (num_joints, 4, 4) global transformation matrices
    """
    num_joints = local_transforms.shape[0]
    global_transforms = np.zeros_like(local_transforms)

    for j in range(num_joints):
        if parents[j] == -1:  # Root joint
            global_transforms[j] = local_transforms[j]
        else:
            parent_idx = int(parents[j])
            global_transforms[j] = global_transforms[parent_idx] @ local_transforms[j]

    return global_transforms


def compute_joint_positions(
    global_transforms: np.ndarray,
) -> np.ndarray:
    """
    Extract joint positions from global transforms.

    Args:
        global_transforms: (num_joints, 4, 4) global transformation matrices

    Returns:
        positions: (num_joints, 3) joint positions in world space
    """
    return global_transforms[:, :3, 3].copy()


def build_local_transforms(
    rotations: np.ndarray,
    t_pose_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """
    Build local transformation matrices from rotations and T-pose.

    Args:
        rotations: (num_joints, 3, 3) local rotation matrices
        t_pose_joints: (num_joints, 3) T-pose joint positions
        parents: (num_joints,) parent joint indices

    Returns:
        local_transforms: (num_joints, 4, 4) local transformation matrices
    """
    num_joints = rotations.shape[0]
    local_transforms = np.zeros((num_joints, 4, 4))

    for j in range(num_joints):
        # Rotation
        local_transforms[j, :3, :3] = rotations[j]

        # Translation (offset from parent in T-pose)
        if parents[j] == -1:
            local_transforms[j, :3, 3] = t_pose_joints[j]
        else:
            parent_idx = int(parents[j])
            local_transforms[j, :3, 3] = t_pose_joints[j] - t_pose_joints[parent_idx]

        local_transforms[j, 3, 3] = 1.0

    return local_transforms


def compute_inverse_bind_matrices(
    t_pose_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """
    Compute inverse bind matrices for skinning.

    The inverse bind matrix transforms vertices from world space to
    joint-local space in the bind pose (T-pose).

    Args:
        t_pose_joints: (num_joints, 3) T-pose joint positions
        parents: (num_joints,) parent joint indices

    Returns:
        inverse_bind_matrices: (num_joints, 4, 4)
    """
    num_joints = t_pose_joints.shape[0]

    # In T-pose, local rotations are identity
    identity_rotations = np.tile(np.eye(3), (num_joints, 1, 1))

    # Build local transforms for T-pose
    local_transforms = build_local_transforms(
        identity_rotations, t_pose_joints, parents
    )

    # Compute global transforms (bind pose)
    global_transforms = forward_kinematics(local_transforms, parents)

    # Inverse of global transforms
    inverse_bind_matrices = np.zeros_like(global_transforms)
    for j in range(num_joints):
        inverse_bind_matrices[j] = np.linalg.inv(global_transforms[j])

    return inverse_bind_matrices


def apply_skinning(
    vertices: np.ndarray,
    joint_transforms: np.ndarray,
    inverse_bind_matrices: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
) -> np.ndarray:
    """
    Apply Linear Blend Skinning (LBS) to deform vertices.

    Args:
        vertices: (V, 3) vertices in bind pose
        joint_transforms: (J, 4, 4) current global joint transforms
        inverse_bind_matrices: (J, 4, 4) inverse bind matrices
        joint_indices: (V, 4) indices of joints affecting each vertex
        joint_weights: (V, 4) weights of joint influence

    Returns:
        deformed_vertices: (V, 3)
    """
    num_vertices = vertices.shape[0]

    # Compute skinning matrices: M_j = G_j @ B_j^{-1}
    skinning_matrices = joint_transforms @ inverse_bind_matrices

    # Add homogeneous coordinate
    vertices_h = np.concatenate([vertices, np.ones((num_vertices, 1))], axis=-1)

    # Apply weighted blend
    deformed_vertices = np.zeros((num_vertices, 3))

    for v in range(num_vertices):
        blended = np.zeros(4)
        for k in range(4):
            j = joint_indices[v, k]
            w = joint_weights[v, k]
            if w > 0:
                blended += w * (skinning_matrices[j] @ vertices_h[v])

        deformed_vertices[v] = blended[:3]

    return deformed_vertices
