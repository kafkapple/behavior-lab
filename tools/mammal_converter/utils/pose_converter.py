"""
Pose conversion utilities: MAMMAL axis-angle → DualPM quaternion+position
"""

import numpy as np
from scipy.spatial.transform import Rotation


def axis_angle_to_quaternion(axis_angle: np.ndarray, output_format: str = "wxyz") -> np.ndarray:
    """
    Convert axis-angle representation to quaternion.

    Args:
        axis_angle: (3,) or (N, 3) axis-angle vectors
        output_format: "wxyz" (DualPM/PyTorch3D) or "xyzw" (scipy)

    Returns:
        quaternion: (4,) or (N, 4)
    """
    single = axis_angle.ndim == 1
    if single:
        axis_angle = axis_angle[None, :]

    quaternions = []
    for aa in axis_angle:
        theta = np.linalg.norm(aa)
        if theta < 1e-8:
            quat = np.array([0., 0., 0., 1.])  # xyzw identity
        else:
            rot = Rotation.from_rotvec(aa)
            quat = rot.as_quat()  # xyzw format

        if output_format == "wxyz":
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])

        quaternions.append(quat)

    quaternions = np.stack(quaternions)
    return quaternions[0] if single else quaternions


def euler_to_rotation_matrix(euler_angles: np.ndarray, order: str = "ZYX") -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.

    Args:
        euler_angles: (3,) Euler angles in radians
        order: Rotation order (default: ZYX for MAMMAL)

    Returns:
        R: (3, 3) rotation matrix
    """
    return Rotation.from_euler(order, euler_angles).as_matrix()


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """
    Rodrigues formula: axis-angle to rotation matrix.

    Args:
        axis_angle: (3,) or (N, 3) axis-angle vectors

    Returns:
        R: (3, 3) or (N, 3, 3) rotation matrices
    """
    single = axis_angle.ndim == 1
    if single:
        axis_angle = axis_angle[None, :]

    batch_size = axis_angle.shape[0]
    R = np.zeros((batch_size, 3, 3))

    for i, aa in enumerate(axis_angle):
        theta = np.linalg.norm(aa)
        if theta < 1e-8:
            R[i] = np.eye(3)
        else:
            R[i] = Rotation.from_rotvec(aa).as_matrix()

    return R[0] if single else R


def convert_mammal_pose_to_dualpm(
    thetas: np.ndarray,
    trans: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    t_pose_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """
    Convert MAMMAL pose parameters to DualPM format.

    Args:
        thetas: (num_joints, 3) local joint rotations (axis-angle)
        trans: (3,) global translation
        rotation: (3,) global rotation (euler ZYX)
        scale: scalar scale factor
        t_pose_joints: (num_joints, 3) T-pose joint positions
        parents: (num_joints,) parent joint indices (-1 for root)

    Returns:
        poses: (num_joints, 7) [quaternion(4), position(3)]
               quaternion in wxyz format
    """
    num_joints = thetas.shape[0]

    # 1. Build local transforms from axis-angle rotations
    local_transforms = np.zeros((num_joints, 4, 4))
    for j in range(num_joints):
        # Local rotation
        R_local = rodrigues(thetas[j])
        local_transforms[j, :3, :3] = R_local

        # Local translation (offset from parent)
        if parents[j] == -1:  # root joint
            local_transforms[j, :3, 3] = t_pose_joints[j]
        else:
            parent_idx = parents[j]
            local_transforms[j, :3, 3] = t_pose_joints[j] - t_pose_joints[parent_idx]

        local_transforms[j, 3, 3] = 1.0

    # 2. Forward kinematics to get global transforms
    global_transforms = np.zeros((num_joints, 4, 4))
    for j in range(num_joints):
        if parents[j] == -1:
            global_transforms[j] = local_transforms[j]
        else:
            parent_idx = parents[j]
            global_transforms[j] = global_transforms[parent_idx] @ local_transforms[j]

    # 3. Apply global transformation (scale, rotation, translation)
    global_R = euler_to_rotation_matrix(rotation, "ZYX")

    for j in range(num_joints):
        # Apply scale to position, then global rotation and translation
        pos = global_transforms[j, :3, 3]
        pos_scaled = scale * pos
        pos_global = global_R @ pos_scaled + trans

        # Apply global rotation to orientation
        R_joint = global_transforms[j, :3, :3]
        R_global = global_R @ R_joint

        global_transforms[j, :3, :3] = R_global
        global_transforms[j, :3, 3] = pos_global

    # 4. Extract quaternion and position
    poses = np.zeros((num_joints, 7))
    for j in range(num_joints):
        # Rotation matrix to quaternion
        R = global_transforms[j, :3, :3]
        quat_xyzw = Rotation.from_matrix(R).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Position
        position = global_transforms[j, :3, 3]

        poses[j, :4] = quat_wxyz
        poses[j, 4:] = position

    return poses


def batch_convert_poses(
    params_list: list,
    t_pose_joints: np.ndarray,
    parents: np.ndarray,
) -> list:
    """
    Convert multiple MAMMAL param files to DualPM poses.

    Args:
        params_list: List of MAMMAL param dicts
        t_pose_joints: (num_joints, 3) T-pose joint positions
        parents: (num_joints,) parent joint indices

    Returns:
        List of pose arrays, each (num_joints, 7)
    """
    poses_list = []
    for params in params_list:
        # Handle batch dimension
        thetas = params['thetas']
        if thetas.ndim == 3:
            thetas = thetas[0]  # Take first batch

        trans = params['trans']
        if trans.ndim == 2:
            trans = trans[0]

        rotation = params['rotation']
        if rotation.ndim == 2:
            rotation = rotation[0]

        scale = params['scale']
        if hasattr(scale, '__len__'):
            scale = float(scale[0] if len(scale.shape) > 0 else scale)

        pose = convert_mammal_pose_to_dualpm(
            thetas=thetas,
            trans=trans,
            rotation=rotation,
            scale=scale,
            t_pose_joints=t_pose_joints,
            parents=parents,
        )
        poses_list.append(pose)

    return poses_list
