"""Keypoint name -> body part. Parts are shared across skeletons; paired parts (ear, forepaw, hindpaw) are
compared without left/right, because the L/R convention of each model was not checked against the others."""
from __future__ import annotations

PAIRED = {"ear": 2, "forepaw": 2, "hindpaw": 2}
PARTS = ("nose", "neck", "ear", "forepaw", "hindpaw", "tailbase", "tailmid", "tailtip")

_SLEAP11 = {"nose1": "nose", "neck1": "neck", "earL1": "ear", "earR1": "ear", "forelegL1": "forepaw",
            "forelegR1": "forepaw", "hindlegL1": "hindpaw", "hindlegR1": "hindpaw", "tailstart1": "tailbase",
            "tail1": "tailmid", "tailend1": "tailtip"}

# ponytail: box models give box centres (arm, leg), not paws; mapped anyway so their distance to paws is visible.
_BOX = {"nose": "nose", "rarm": "forepaw", "larm": "forepaw", "rleg": "hindpaw", "lleg": "hindpaw", "tail": "tailtip"}

_SA_QUADRUPED = {"nose": "nose", "neck_base": "neck", "right_earend": "ear", "left_earend": "ear",
                 "front_left_paw": "forepaw", "front_right_paw": "forepaw", "back_left_paw": "hindpaw",
                 "back_right_paw": "hindpaw", "tail_base": "tailbase", "tail_end": "tailtip"}

# names follow DeepLabCut's SuperAnimal-TopViewMouse model card; check against the run output (compare raises otherwise).
_SA_TOPVIEWMOUSE = {"nose": "nose", "neck": "neck", "left_ear": "ear", "right_ear": "ear",
                    "tail_base": "tailbase", "tail4": "tailmid", "tail_end": "tailtip"}

# AP-10K order of the COCO-style 17 keypoints (checked against the model config at run time by run_vitpose).
_VITPOSE_AP10K = {"nose": "nose", "neck": "neck", "root_of_tail": "tailbase", "left_front_paw": "forepaw",
                  "right_front_paw": "forepaw", "left_back_paw": "hindpaw", "right_back_paw": "hindpaw"}

PART_MAP: dict[str, dict[str, str]] = {
    "sleap_1423": _SLEAP11,
    "sleap_tailless_1501": {k: v for k, v in _SLEAP11.items() if k not in ("tail1", "tailend1")},
    "rtdetr": _BOX, "yolo_avatar3d_train": _BOX, "yolo_avatar3d_balbc": _BOX, "yolo_khu_527": _BOX,
    "superanimal_quadruped": _SA_QUADRUPED,
    "superanimal_topviewmouse": _SA_TOPVIEWMOUSE,
    "vitpose_plus_ap10k": _VITPOSE_AP10K,
}
