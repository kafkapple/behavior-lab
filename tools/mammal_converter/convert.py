#!/usr/bin/env python3
"""
MAMMAL_mouse to DualPM Dataset Converter

Converts MAMMAL_mouse fitting results to DualPM training dataset format.

Usage:
    python convert.py \
        --mammal_dir /path/to/MAMMAL_mouse \
        --fitting_result fitting_result_name \
        --output_dir /path/to/output
"""

import argparse
import pickle
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import (
    convert_mammal_pose_to_dualpm,
    build_gltf_with_skeleton,
)


def load_mammal_model(mammal_dir: Path) -> dict:
    """
    Load MAMMAL mouse model data.

    Returns dict with:
        - vertices: (V, 3) T-pose vertices
        - faces: (F, 3) triangle indices
        - skinning_weights: (V, J) weights
        - t_pose_joints: (J, 3) T-pose joint positions
        - parents: (J,) parent indices
        - joint_names: list of joint names
    """
    model_dir = mammal_dir / "mouse_model"
    txt_dir = model_dir / "mouse_txt"

    # Load pickle files
    with open(txt_dir / "parents.pkl", 'rb') as f:
        parents = np.array(pickle.load(f))

    with open(txt_dir / "id_to_names.pkl", 'rb') as f:
        joint_names = pickle.load(f)

    with open(txt_dir / "init_joint_trans.pkl", 'rb') as f:
        t_pose_joints = np.array(pickle.load(f))

    # Load text files
    vertices = np.loadtxt(txt_dir / "vertices.txt")
    faces = np.loadtxt(txt_dir / "faces_vert.txt", dtype=np.int64)

    # Load skinning weights (sparse format)
    weights_data = np.loadtxt(txt_dir / "skinning_weights.txt")
    num_vertices = vertices.shape[0]
    num_joints = len(joint_names)

    skinning_weights = np.zeros((num_vertices, num_joints))
    for row in weights_data:
        joint_id = int(row[0])
        vertex_id = int(row[1])
        weight = row[2]
        skinning_weights[vertex_id, joint_id] = weight

    return {
        'vertices': vertices,
        'faces': faces,
        'skinning_weights': skinning_weights,
        't_pose_joints': t_pose_joints,
        'parents': parents,
        'joint_names': joint_names,
    }


def load_fitting_params(fitting_dir: Path) -> list[dict]:
    """Load all param files from fitting results."""
    params_dir = fitting_dir / "params"
    param_files = sorted(params_dir.glob("param*.pkl"))

    # Filter out silhouette-refined params if base params exist
    base_params = [f for f in param_files if "_sil" not in f.name]

    params_list = []
    for param_file in tqdm(base_params, desc="Loading params"):
        with open(param_file, 'rb') as f:
            params = pickle.load(f)

        # Convert torch tensors to numpy if needed
        converted = {}
        for key, value in params.items():
            if hasattr(value, 'cpu'):
                # Handle tensors with requires_grad
                if hasattr(value, 'detach'):
                    converted[key] = value.detach().cpu().numpy()
                else:
                    converted[key] = value.cpu().numpy()
            else:
                converted[key] = np.array(value)

        params_list.append(converted)

    return params_list


def load_camera_params(mammal_dir: Path, dataset_name: str, view_id: int = 0) -> dict:
    """Load camera parameters from MAMMAL dataset."""
    data_dir = mammal_dir / "data" / "examples" / dataset_name

    # Try to find camera calibration
    calib_file = data_dir / "calibration.json"
    if calib_file.exists():
        with open(calib_file) as f:
            calib = json.load(f)

        # Extract intrinsics and extrinsics for specified view
        camera = calib.get(f"cam{view_id}", calib.get(str(view_id), {}))

        return {
            'intrinsic': np.array(camera.get('K', np.eye(3))),
            'extrinsic': np.array(camera.get('RT', np.eye(4))),
            'distortion': np.array(camera.get('dist', np.zeros(5))),
        }

    # Fallback: return identity/default camera
    return {
        'intrinsic': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
        'extrinsic': np.eye(4),
        'distortion': np.zeros(5),
    }


def load_images(mammal_dir: Path, dataset_name: str, view_id: int = 0) -> list[tuple]:
    """
    Load RGB images and masks from MAMMAL dataset.

    Returns list of (frame_id, rgb_path, mask_path)
    """
    data_dir = mammal_dir / "data" / "examples" / dataset_name

    # Find image directory for the view
    view_dir = data_dir / f"cam{view_id}"
    if not view_dir.exists():
        view_dir = data_dir / f"view{view_id}"
    if not view_dir.exists():
        view_dir = data_dir  # Single view case

    # Find images
    image_files = sorted(view_dir.glob("*.png")) + sorted(view_dir.glob("*.jpg"))

    results = []
    for img_path in image_files:
        frame_id = img_path.stem
        # Try to find corresponding mask
        mask_path = view_dir / f"{frame_id}_mask.png"
        if not mask_path.exists():
            mask_path = view_dir / "masks" / f"{frame_id}.png"
        if not mask_path.exists():
            mask_path = None

        results.append((frame_id, img_path, mask_path))

    return results


def create_dualpm_dataset(
    mammal_dir: Path,
    fitting_result: str,
    output_dir: Path,
    resolution: int = 160,
    train_split: float = 0.9,
    camera_view: int = 0,
) -> None:
    """
    Create DualPM dataset from MAMMAL fitting results.

    Args:
        mammal_dir: Path to MAMMAL_mouse project
        fitting_result: Name of fitting result folder
        output_dir: Output directory for DualPM dataset
        resolution: Output image resolution
        train_split: Fraction of data for training
        camera_view: Camera view to use (0-5)
    """
    print(f"Converting MAMMAL_mouse to DualPM dataset")
    print(f"  MAMMAL dir: {mammal_dir}")
    print(f"  Fitting result: {fitting_result}")
    print(f"  Output dir: {output_dir}")

    # Create output directories
    output_dir = Path(output_dir)
    (output_dir / "shapes" / "mouse").mkdir(parents=True, exist_ok=True)
    (output_dir / "poses").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "renders").mkdir(parents=True, exist_ok=True)
    (output_dir / "cameras").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    # 1. Load MAMMAL model
    print("\n[1/5] Loading MAMMAL model...")
    model_data = load_mammal_model(mammal_dir)

    # 2. Build GLTF template mesh with skeleton
    print("\n[2/5] Building GLTF template...")
    gltf_path = output_dir / "shapes" / "mouse" / "mouse_shape.gltf"
    build_gltf_with_skeleton(
        vertices=model_data['vertices'],
        faces=model_data['faces'],
        joint_names=model_data['joint_names'],
        parents=model_data['parents'],
        t_pose_joints=model_data['t_pose_joints'],
        skinning_weights=model_data['skinning_weights'],
        output_path=gltf_path,
        model_name="mouse",
    )
    print(f"  Saved: {gltf_path}")

    # 3. Load and convert poses
    print("\n[3/5] Converting poses...")
    fitting_dir = mammal_dir / "results" / "fitting" / fitting_result
    params_list = load_fitting_params(fitting_dir)

    frame_ids = []
    for i, params in enumerate(tqdm(params_list, desc="Converting poses")):
        frame_id = f"{i:06d}"
        frame_ids.append(frame_id)

        # Get pose parameters
        thetas = params['thetas']
        if thetas.ndim == 3:
            thetas = thetas[0]

        trans = params['trans']
        if trans.ndim == 2:
            trans = trans[0]

        rotation = params['rotation']
        if rotation.ndim == 2:
            rotation = rotation[0]

        scale = params['scale']
        if hasattr(scale, '__len__'):
            scale = float(np.asarray(scale).flatten()[0])
        else:
            scale = float(scale)

        # Convert to DualPM format
        pose = convert_mammal_pose_to_dualpm(
            thetas=thetas,
            trans=trans,
            rotation=rotation,
            scale=scale,
            t_pose_joints=model_data['t_pose_joints'],
            parents=model_data['parents'],
        )

        # Save as NPZ
        pose_path = output_dir / "poses" / f"{frame_id}_pose.npz"
        np.savez(pose_path, poses=pose)

    print(f"  Converted {len(params_list)} poses")

    # 4. Copy/resize images and masks
    print("\n[4/5] Processing images...")

    # Get dataset name from fitting result
    dataset_name = "_".join(fitting_result.split("_")[:-1])  # Remove timestamp

    # Try to find source images
    fitting_render_dir = fitting_dir / "render"
    if fitting_render_dir.exists():
        # Use fitting renders
        render_files = sorted(fitting_render_dir.glob("fitting_*.png"))
        for i, render_path in enumerate(tqdm(render_files, desc="Processing renders")):
            if i >= len(frame_ids):
                break
            frame_id = frame_ids[i]

            # Load and resize
            img = Image.open(render_path)
            img_resized = img.resize((resolution, resolution), Image.LANCZOS)

            # Save RGB
            rgb_path = output_dir / "renders" / f"{frame_id}_rgb.png"
            img_resized.save(rgb_path)

            # Generate simple mask from alpha or create placeholder
            if img.mode == 'RGBA':
                alpha = img.split()[3]
                mask = alpha.point(lambda x: 255 if x > 128 else 0)
            else:
                # Create full mask
                mask = Image.new('L', img.size, 255)

            mask_resized = mask.resize((resolution, resolution), Image.NEAREST)
            mask_path = output_dir / "masks" / f"{frame_id}_mask.png"
            mask_resized.save(mask_path)

    # 5. Create camera and metadata files
    print("\n[5/5] Creating metadata...")

    # Try to load camera params
    camera_params = load_camera_params(mammal_dir, dataset_name, camera_view)

    for frame_id in tqdm(frame_ids, desc="Creating metadata"):
        # Camera file (view matrix format)
        camera_path = output_dir / "cameras" / f"{frame_id}_camera.txt"
        view_matrix = camera_params['extrinsic']
        with open(camera_path, 'w') as f:
            for row in view_matrix:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        # Metadata file
        meta_path = output_dir / "metadata" / f"{frame_id}_metadata.txt"
        with open(meta_path, 'w') as f:
            f.write(f"model_name: mouse\n")
            f.write(f"focal_length: {camera_params['intrinsic'][0, 0]:.2f}\n")
            f.write(f"frame_id: {frame_id}\n")

    # Create train/val split
    num_frames = len(frame_ids)
    num_train = int(num_frames * train_split)

    np.random.seed(42)
    shuffled_ids = np.random.permutation(frame_ids)
    train_ids = shuffled_ids[:num_train]
    val_ids = shuffled_ids[num_train:]

    # Save benchmark file
    benchmark_path = output_dir / "train_benchmark.txt"
    with open(benchmark_path, 'w') as f:
        f.write("# train\n")
        for fid in sorted(train_ids):
            f.write(f"{fid}\n")
        f.write("# val\n")
        for fid in sorted(val_ids):
            f.write(f"{fid}\n")

    print(f"\n✅ Dataset created at: {output_dir}")
    print(f"   Total frames: {num_frames}")
    print(f"   Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"\nNext steps:")
    print(f"  1. Extract DINOv2 features (optional)")
    print(f"  2. Run DualPM training:")
    print(f"     python scripts/train.py dataset_root={output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MAMMAL_mouse fitting results to DualPM dataset"
    )
    parser.add_argument(
        "--mammal_dir", "-m",
        type=Path,
        required=True,
        help="Path to MAMMAL_mouse project directory"
    )
    parser.add_argument(
        "--fitting_result", "-f",
        type=str,
        required=True,
        help="Name of fitting result folder (in results/fitting/)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="Output directory for DualPM dataset"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=160,
        help="Output image resolution (default: 160)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)"
    )
    parser.add_argument(
        "--camera_view",
        type=int,
        default=0,
        help="Camera view to use, 0-5 (default: 0)"
    )

    args = parser.parse_args()

    create_dualpm_dataset(
        mammal_dir=args.mammal_dir,
        fitting_result=args.fitting_result,
        output_dir=args.output_dir,
        resolution=args.resolution,
        train_split=args.train_split,
        camera_view=args.camera_view,
    )


if __name__ == "__main__":
    main()
