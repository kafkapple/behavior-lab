"""
GLTF 2.0 builder for rigged meshes with skeleton.
"""

import numpy as np
import json
import base64
import struct
from pathlib import Path
from typing import Optional

from .forward_kinematics import compute_inverse_bind_matrices


def build_gltf_with_skeleton(
    vertices: np.ndarray,
    faces: np.ndarray,
    joint_names: list[str],
    parents: np.ndarray,
    t_pose_joints: np.ndarray,
    skinning_weights: np.ndarray,
    output_path: Path,
    model_name: str = "mouse",
) -> None:
    """
    Build a GLTF 2.0 file with rigged mesh and skeleton.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle indices
        joint_names: List of joint names
        parents: (J,) parent joint indices (-1 for root)
        t_pose_joints: (J, 3) T-pose joint positions
        skinning_weights: (V, J) skinning weights per vertex per joint
        output_path: Path to save GLTF file
        model_name: Name for the model
    """
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    num_joints = len(joint_names)

    # Compute inverse bind matrices
    inverse_bind_matrices = compute_inverse_bind_matrices(t_pose_joints, parents)

    # Compute joint indices and weights per vertex (top 4)
    joint_indices, joint_weights = _compute_vertex_joint_weights(
        skinning_weights, max_influences=4
    )

    # Build binary buffer
    buffer_data = bytearray()

    # 1. Vertex positions (POSITION)
    positions_offset = len(buffer_data)
    positions_data = vertices.astype(np.float32).tobytes()
    buffer_data.extend(positions_data)
    positions_length = len(positions_data)

    # 2. Face indices (INDICES)
    # Pad to 4-byte alignment
    while len(buffer_data) % 4 != 0:
        buffer_data.append(0)
    indices_offset = len(buffer_data)
    indices_data = faces.astype(np.uint32).flatten().tobytes()
    buffer_data.extend(indices_data)
    indices_length = len(indices_data)

    # 3. Joint indices (JOINTS_0) - uint8 or uint16
    while len(buffer_data) % 4 != 0:
        buffer_data.append(0)
    joints_offset = len(buffer_data)
    if num_joints <= 255:
        joints_data = joint_indices.astype(np.uint8).tobytes()
        joints_component_type = 5121  # UNSIGNED_BYTE
    else:
        joints_data = joint_indices.astype(np.uint16).tobytes()
        joints_component_type = 5123  # UNSIGNED_SHORT
    buffer_data.extend(joints_data)
    joints_length = len(joints_data)

    # 4. Joint weights (WEIGHTS_0)
    while len(buffer_data) % 4 != 0:
        buffer_data.append(0)
    weights_offset = len(buffer_data)
    weights_data = joint_weights.astype(np.float32).tobytes()
    buffer_data.extend(weights_data)
    weights_length = len(weights_data)

    # 5. Inverse bind matrices
    while len(buffer_data) % 4 != 0:
        buffer_data.append(0)
    ibm_offset = len(buffer_data)
    # GLTF uses column-major, transpose for correct storage
    ibm_col_major = inverse_bind_matrices.transpose(0, 2, 1).astype(np.float32)
    ibm_data = ibm_col_major.tobytes()
    buffer_data.extend(ibm_data)
    ibm_length = len(ibm_data)

    # Encode buffer as base64
    buffer_base64 = base64.b64encode(buffer_data).decode('ascii')
    buffer_uri = f"data:application/octet-stream;base64,{buffer_base64}"

    # Compute bounding box
    pos_min = vertices.min(axis=0).tolist()
    pos_max = vertices.max(axis=0).tolist()

    # Build GLTF structure
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "MAMMAL_to_DualPM_Converter"
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [],
        "meshes": [{
            "name": model_name,
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "JOINTS_0": 1,
                    "WEIGHTS_0": 2
                },
                "indices": 3,
                "mode": 4  # TRIANGLES
            }]
        }],
        "skins": [{
            "joints": list(range(num_joints)),
            "inverseBindMatrices": 4,
            "skeleton": _find_root_joint(parents)
        }],
        "accessors": [
            # 0: POSITION
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": num_vertices,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max
            },
            # 1: JOINTS_0
            {
                "bufferView": 1,
                "componentType": joints_component_type,
                "count": num_vertices,
                "type": "VEC4"
            },
            # 2: WEIGHTS_0
            {
                "bufferView": 2,
                "componentType": 5126,  # FLOAT
                "count": num_vertices,
                "type": "VEC4"
            },
            # 3: INDICES
            {
                "bufferView": 3,
                "componentType": 5125,  # UNSIGNED_INT
                "count": num_faces * 3,
                "type": "SCALAR"
            },
            # 4: Inverse Bind Matrices
            {
                "bufferView": 4,
                "componentType": 5126,  # FLOAT
                "count": num_joints,
                "type": "MAT4"
            }
        ],
        "bufferViews": [
            # 0: POSITION
            {"buffer": 0, "byteOffset": positions_offset, "byteLength": positions_length},
            # 1: JOINTS_0
            {"buffer": 0, "byteOffset": joints_offset, "byteLength": joints_length},
            # 2: WEIGHTS_0
            {"buffer": 0, "byteOffset": weights_offset, "byteLength": weights_length},
            # 3: INDICES
            {"buffer": 0, "byteOffset": indices_offset, "byteLength": indices_length},
            # 4: IBM
            {"buffer": 0, "byteOffset": ibm_offset, "byteLength": ibm_length}
        ],
        "buffers": [{
            "byteLength": len(buffer_data),
            "uri": buffer_uri
        }]
    }

    # Build node hierarchy for skeleton
    nodes = _build_skeleton_nodes(joint_names, parents, t_pose_joints)

    # Add mesh node (skinned)
    mesh_node = {
        "name": f"{model_name}_mesh",
        "mesh": 0,
        "skin": 0
    }
    nodes.insert(0, mesh_node)

    # Update joint indices in skin (offset by 1 due to mesh node)
    gltf["skins"][0]["joints"] = list(range(1, num_joints + 1))
    gltf["skins"][0]["skeleton"] = _find_root_joint(parents) + 1

    # Update scene root
    gltf["scenes"][0]["nodes"] = [0, _find_root_joint(parents) + 1]

    gltf["nodes"] = nodes

    # Save GLTF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(gltf, f, indent=2)


def _compute_vertex_joint_weights(
    skinning_weights: np.ndarray,
    max_influences: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert full skinning weight matrix to top-k joint indices and weights.

    Args:
        skinning_weights: (V, J) weight matrix
        max_influences: Maximum number of joint influences per vertex

    Returns:
        joint_indices: (V, max_influences) int
        joint_weights: (V, max_influences) float, normalized
    """
    num_vertices, num_joints = skinning_weights.shape

    joint_indices = np.zeros((num_vertices, max_influences), dtype=np.int32)
    joint_weights = np.zeros((num_vertices, max_influences), dtype=np.float32)

    for v in range(num_vertices):
        weights = skinning_weights[v]
        # Get top k joints by weight
        top_k_indices = np.argsort(weights)[-max_influences:][::-1]

        for k, j in enumerate(top_k_indices):
            joint_indices[v, k] = j
            joint_weights[v, k] = weights[j]

        # Normalize weights to sum to 1
        weight_sum = joint_weights[v].sum()
        if weight_sum > 0:
            joint_weights[v] /= weight_sum

    return joint_indices, joint_weights


def _find_root_joint(parents: np.ndarray) -> int:
    """Find the root joint index (parent == -1)."""
    for j, p in enumerate(parents):
        if p == -1:
            return j
    return 0


def _build_skeleton_nodes(
    joint_names: list[str],
    parents: np.ndarray,
    t_pose_joints: np.ndarray,
) -> list[dict]:
    """
    Build GLTF node hierarchy for skeleton.

    Args:
        joint_names: List of joint names
        parents: (J,) parent indices
        t_pose_joints: (J, 3) T-pose joint positions

    Returns:
        List of GLTF node dictionaries
    """
    num_joints = len(joint_names)
    nodes = []

    # Build children lists
    children = {j: [] for j in range(num_joints)}
    for j, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(j)

    for j in range(num_joints):
        node = {"name": joint_names[j]}

        # Translation relative to parent
        if parents[j] == -1:
            translation = t_pose_joints[j].tolist()
        else:
            parent_idx = int(parents[j])
            translation = (t_pose_joints[j] - t_pose_joints[parent_idx]).tolist()

        node["translation"] = translation

        # Add children
        if children[j]:
            # Offset by 1 due to mesh node at index 0
            node["children"] = [c + 1 for c in children[j]]

        nodes.append(node)

    return nodes
