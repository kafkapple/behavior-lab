"""Every 2D keypoint predictor considered for the AVATAR data, with what can be said about it today."""
from __future__ import annotations

from dataclasses import dataclass

RAN, ZERO_SHOT, NEEDS_LABELS, SKIPPED = "ran", "zero-shot runnable", "needs hand labels", "skipped"


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    n_keypoints: int | None
    status: str
    note: str


MODELS: tuple[Spec, ...] = (
    Spec("sleap_1423", "SLEAP single-instance", 11, RAN, "SUBTLE weights n=1423; label set of the GT template"),
    Spec("sleap_tailless_1501", "SLEAP single-instance", 9, RAN, "SUBTLE weights n=1501, no tail1/tailend1"),
    Spec("yolo_avatar3d_train", "YOLO11m box", 9, RAN, "box centres per body part, not keypoints"),
    Spec("yolo_avatar3d_balbc", "YOLO11m box", 9, RAN, "box centres per body part"),
    Spec("yolo_khu_527", "YOLO11m box", 9, RAN, "box centres per body part"),
    Spec("rtdetr", "RT-DETRv2 box", 9, RAN, "box centres per body part"),
    Spec("superanimal_quadruped", "DLC SuperAnimal", 39, RAN, "HRNet-w32 + Faster R-CNN, zero-shot"),
    Spec("superanimal_topviewmouse", "DLC SuperAnimal", 27, ZERO_SHOT, "top-view training data; one AVATAR camera is a bottom view"),
    Spec("vitpose_plus_ap10k", "ViTPose++ (transformers)", 17, ZERO_SHOT, "AP-10K expert (dataset_index 3), needs boxes"),
    Spec("lightning_pose", "Lightning Pose (+EKS)", None, NEEDS_LABELS, "semi-supervised; trains on the labelled frames"),
    Spec("dlc_resnet", "DeepLabCut own training", None, NEEDS_LABELS, "trains on the labelled frames"),
    Spec("sleap_retrain", "SLEAP retrain", None, NEEDS_LABELS, "training package not reachable yet"),
    Spec("dannce", "DANNCE / s-DANNCE", None, NEEDS_LABELS, "needs 3D labels and calibration (label3d), not 2D"),
    Spec("rtmpose_mmpose", "MMPose / RTMPose", None, SKIPPED, "animal checkpoints are AP-10K trained, same domain as ViTPose++ AP-10K"),
    Spec("yolo_pose_human", "YOLO-pose (COCO person)", 17, SKIPPED, "human skeleton, no mouse prior"),
)

_BY_NAME = {m.name: m for m in MODELS}


def get(name: str) -> Spec:
    return _BY_NAME[name]


def by_status(status: str) -> list[Spec]:
    return [m for m in MODELS if m.status == status]
