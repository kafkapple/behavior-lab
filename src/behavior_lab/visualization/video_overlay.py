"""OpenCV-based keypoint overlay on video frames.

Renders skeleton joints and limbs directly onto video frames,
producing annotated video/GIF output. Uses the same semantic
color system as skeleton.py (BODY_PART_COLORS).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from behavior_lab.core.skeleton import SkeletonDefinition
from behavior_lab.visualization.colors import (
    get_joint_colors,
    get_limb_colors,
    get_person_colors,
    get_joint_labels,
    hex_to_bgr,
)


def render_skeleton_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    skeleton: SkeletonDefinition,
    *,
    joint_radius: int = 4,
    limb_thickness: int = 2,
    joint_colors_bgr: list[tuple[int, int, int]] | None = None,
    limb_colors_bgr: list[tuple[int, int, int]] | None = None,
    show_labels: bool = False,
    label_font_scale: float = 0.35,
    opacity: float = 0.8,
) -> np.ndarray:
    """Draw skeleton overlay on a single BGR frame.

    Args:
        frame: (H, W, 3) BGR uint8 image.
        keypoints: (K, 2) or (K*num_persons, 2) pixel coordinates.
        skeleton: SkeletonDefinition providing edges, body_parts, num_persons.
        joint_radius: Radius in pixels for joint circles.
        limb_thickness: Line thickness in pixels.
        joint_colors_bgr: Pre-computed BGR tuples per joint. Auto-derived if None.
        limb_colors_bgr: Pre-computed BGR tuples per edge. Auto-derived if None.
        show_labels: Render joint abbreviation near each joint.
        label_font_scale: Font scale for cv2.putText.
        opacity: Alpha blending factor for overlay (0.0-1.0).

    Returns:
        Annotated frame (H, W, 3) BGR uint8.
    """
    import cv2

    K = keypoints.shape[0]
    jpn = skeleton.num_joints
    num_persons = max(1, K // jpn) if K > jpn else 1

    # Auto-derive colors
    if joint_colors_bgr is None:
        if num_persons > 1:
            person_hex = get_person_colors(num_persons)
            joint_colors_bgr = []
            for p in range(num_persons):
                jc = get_joint_colors(skeleton)
                joint_colors_bgr.extend([hex_to_bgr(c) for c in jc])
        else:
            jc = get_joint_colors(skeleton)
            joint_colors_bgr = [hex_to_bgr(c) for c in jc]

    if limb_colors_bgr is None:
        if num_persons > 1:
            limb_colors_bgr = []
            for p in range(num_persons):
                lc = get_limb_colors(skeleton)
                limb_colors_bgr.extend([hex_to_bgr(c) for c in lc])
        else:
            lc = get_limb_colors(skeleton)
            limb_colors_bgr = [hex_to_bgr(c) for c in lc]

    labels = get_joint_labels(skeleton) if show_labels else None

    # Draw on overlay for alpha blending
    overlay = frame.copy()

    for p in range(num_persons):
        start = p * jpn
        end = start + jpn
        pkp = keypoints[start:end]

        # Skip zero-valued joints
        nonzero = np.any(pkp != 0, axis=-1)

        # Draw limbs first (behind joints)
        for ei, (i, j) in enumerate(skeleton.edges):
            if i < jpn and j < jpn and nonzero[i] and nonzero[j]:
                pt1 = (int(round(pkp[i, 0])), int(round(pkp[i, 1])))
                pt2 = (int(round(pkp[j, 0])), int(round(pkp[j, 1])))
                ec_idx = p * len(skeleton.edges) + ei
                ec = limb_colors_bgr[ec_idx] if ec_idx < len(limb_colors_bgr) else (128, 128, 128)
                cv2.line(overlay, pt1, pt2, ec, limb_thickness, cv2.LINE_AA)

        # Draw joints
        for ji in range(jpn):
            if not nonzero[ji]:
                continue
            pt = (int(round(pkp[ji, 0])), int(round(pkp[ji, 1])))
            jc_idx = start + ji
            jc = joint_colors_bgr[jc_idx] if jc_idx < len(joint_colors_bgr) else (128, 128, 128)
            cv2.circle(overlay, pt, joint_radius, jc, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, joint_radius, (255, 255, 255), 1, cv2.LINE_AA)

            if labels and ji < len(labels):
                cv2.putText(
                    overlay, labels[ji],
                    (pt[0] + joint_radius + 2, pt[1] - joint_radius),
                    cv2.FONT_HERSHEY_SIMPLEX, label_font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

    # Alpha blend
    if opacity < 1.0:
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        return frame
    return overlay


def overlay_keypoints_on_video(
    video_path: str | Path,
    keypoints: np.ndarray,
    skeleton: SkeletonDefinition,
    output_path: str | Path,
    *,
    fps: float | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int | None = None,
    joint_radius: int = 4,
    limb_thickness: int = 2,
    show_labels: bool = False,
    opacity: float = 0.8,
    output_format: Literal["mp4", "gif"] = "gif",
    resize_width: int | None = None,
) -> Path:
    """Overlay skeleton keypoints on video frames and write annotated output.

    Args:
        video_path: Path to input video file.
        keypoints: (T, K, 2) pixel-coordinate keypoints.
        skeleton: SkeletonDefinition.
        output_path: Path for output file.
        fps: Output FPS. If None, uses input video FPS.
        start_frame: First frame to process.
        end_frame: Last frame (exclusive). None = min(video, keypoints).
        max_frames: Maximum frames to render. None = no limit.
        joint_radius: Pixel radius for joints.
        limb_thickness: Line thickness for limbs.
        show_labels: Render joint abbreviation text.
        opacity: Alpha blending (0.0-1.0).
        output_format: "mp4" or "gif".
        resize_width: Resize output width (maintains aspect ratio). None = original.

    Returns:
        Path to the generated output file.
    """
    import cv2

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps is None:
        fps = video_fps

    # Determine frame range
    kp_frames = keypoints.shape[0]
    actual_end = min(total_video_frames, kp_frames + start_frame)
    if end_frame is not None:
        actual_end = min(actual_end, end_frame)
    n_frames = actual_end - start_frame
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    # Resize calculation
    if resize_width is not None and resize_width != width:
        scale = resize_width / width
        out_w = resize_width
        out_h = int(height * scale)
    else:
        scale = 1.0
        out_w = width
        out_h = height

    # Pre-compute colors for efficiency
    jpn = skeleton.num_joints
    K = keypoints.shape[1]
    num_persons = max(1, K // jpn) if K > jpn else 1

    jc_bgr = []
    lc_bgr = []
    for p in range(num_persons):
        jc_hex = get_joint_colors(skeleton)
        lc_hex = get_limb_colors(skeleton)
        jc_bgr.extend([hex_to_bgr(c) for c in jc_hex])
        lc_bgr.extend([hex_to_bgr(c) for c in lc_hex])

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if output_format == "gif":
        frames_out = []

    elif output_format == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        kp_idx = fi  # keypoints[0] = video[start_frame]
        if kp_idx >= kp_frames:
            break

        kp = keypoints[kp_idx].copy()

        # Scale keypoints if resizing
        if scale != 1.0:
            kp = kp * scale
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        annotated = render_skeleton_on_frame(
            frame, kp, skeleton,
            joint_radius=joint_radius,
            limb_thickness=limb_thickness,
            joint_colors_bgr=jc_bgr,
            limb_colors_bgr=lc_bgr,
            show_labels=show_labels,
            opacity=opacity,
        )

        if output_format == "gif":
            # Convert BGR to RGB for PIL/imageio
            frames_out.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        elif output_format == "mp4":
            writer.write(annotated)

    cap.release()

    if output_format == "gif":
        import imageio.v3 as iio
        duration_ms = int(1000 / fps)
        iio.imwrite(
            str(output_path),
            frames_out,
            extension=".gif",
            duration=duration_ms,
            loop=0,
        )
    elif output_format == "mp4":
        writer.release()

    return output_path


def overlay_keypoints_on_frame_array(
    frames: np.ndarray | list[np.ndarray],
    keypoints: np.ndarray,
    skeleton: SkeletonDefinition,
    output_path: str | Path,
    *,
    fps: float = 15.0,
    max_frames: int | None = None,
    joint_radius: int = 4,
    limb_thickness: int = 2,
    show_labels: bool = False,
    opacity: float = 0.8,
    output_format: Literal["mp4", "gif"] = "gif",
) -> Path:
    """Overlay keypoints on a sequence of frame arrays (no video file needed).

    Useful when frames are extracted from numpy arrays or loaded in memory.

    Args:
        frames: (T, H, W, 3) RGB uint8 array or list of such frames.
        keypoints: (T, K, 2) pixel-coordinate keypoints.
        skeleton: SkeletonDefinition.
        output_path: Output file path.
        fps: Output FPS.
        max_frames: Cap on frames to render.
        joint_radius: Joint circle radius.
        limb_thickness: Limb line thickness.
        show_labels: Show joint labels.
        opacity: Alpha blending.
        output_format: "mp4" or "gif".

    Returns:
        Path to generated file.
    """
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(frames), keypoints.shape[0])
    if max_frames is not None:
        n = min(n, max_frames)

    # Pre-compute colors
    jpn = skeleton.num_joints
    K = keypoints.shape[1]
    num_persons = max(1, K // jpn) if K > jpn else 1
    jc_bgr = []
    lc_bgr = []
    for p in range(num_persons):
        jc_bgr.extend([hex_to_bgr(c) for c in get_joint_colors(skeleton)])
        lc_bgr.extend([hex_to_bgr(c) for c in get_limb_colors(skeleton)])

    annotated_frames = []
    for fi in range(n):
        f = frames[fi]
        # Ensure BGR for OpenCV
        if isinstance(f, np.ndarray) and f.ndim == 3 and f.shape[2] == 3:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.dtype == np.uint8 else f
        else:
            bgr = f

        result = render_skeleton_on_frame(
            bgr, keypoints[fi], skeleton,
            joint_radius=joint_radius,
            limb_thickness=limb_thickness,
            joint_colors_bgr=jc_bgr,
            limb_colors_bgr=lc_bgr,
            show_labels=show_labels,
            opacity=opacity,
        )
        # Back to RGB for output
        annotated_frames.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    if output_format == "gif":
        import imageio.v3 as iio
        duration_ms = int(1000 / fps)
        iio.imwrite(
            str(output_path), annotated_frames,
            extension=".gif", duration=duration_ms, loop=0,
        )
    elif output_format == "mp4":
        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for f in annotated_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

    return output_path
