"""ViTPose++ (transformers) top-down keypoints with the AP-10K expert, run on the GPU host.

Standalone (needs torch, transformers, pillow, pandas):
    python run_vitpose.py --images-root <dir> --boxes boxes.csv --out vitpose_plus_ap10k.csv

`boxes.csv`: image,frame,camera,x0,y0,x1,y1 (pixels of that image). Top-down models need a box; `boxes_from_table`
builds them from another model's points, so this run measures keypoints inside that box and not detection.
Writes the long prediction table (`tables.COLUMNS`) with the AP-10K keypoint names.
"""
from __future__ import annotations

import argparse
from pathlib import Path

CHECKPOINT = "usyd-community/vitpose-plus-base"
AP10K_EXPERT = 3  # ViTPose++ MoE heads: COCO 0, AiC 1, MPII 2, AP-10K 3, APT-36K 4, COCO-WholeBody 5 (model card)
# Output channel order of the AP-10K expert (mmpose configs/_base_/datasets/ap10k.py). The checkpoint's id2label lists the
# COCO-17 names (Nose, L_Eye, ...), which are the wrong order for this expert; using them swaps eyes, nose and neck.
AP10K_NAMES = ["left_eye", "right_eye", "nose", "neck", "root_of_tail", "left_shoulder", "left_elbow", "left_front_paw",
               "right_shoulder", "right_elbow", "right_front_paw", "left_hip", "left_knee", "left_back_paw",
               "right_hip", "right_knee", "right_back_paw"]


def boxes_from_table(df, model: str, pad: float = 0.25) -> "pd.DataFrame":
    """One box per (frame, camera) from the confident points of `model`, padded by `pad` of its size per side."""
    import pandas as pd

    rows = []
    sub = df[(df["model"] == model) & df["x_px"].notna()]
    for (frame, camera), g in sub.groupby(["frame", "camera"]):
        x0, x1, y0, y1 = g["x_px"].min(), g["x_px"].max(), g["y_px"].min(), g["y_px"].max()
        w, h = max(x1 - x0, 20.0), max(y1 - y0, 20.0)
        rows.append({"frame": frame, "camera": camera, "x0": x0 - pad * w, "y0": y0 - pad * h,
                     "x1": x1 + pad * w, "y1": y1 + pad * h})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="vitpose_plus_ap10k")
    args = ap.parse_args()
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model = VitPoseForPoseEstimation.from_pretrained(CHECKPOINT).to(device).eval()
    names = AP10K_NAMES
    boxes = pd.read_csv(args.boxes)
    rows = []
    for rec in boxes.itertuples(index=False):
        image = Image.open(Path(args.images_root) / rec.image).convert("RGB")
        box = [[rec.x0, rec.y0, rec.x1 - rec.x0, rec.y1 - rec.y0]]  # COCO xywh
        inputs = processor(image, boxes=[box], return_tensors="pt").to(device)
        dataset_index = torch.tensor([AP10K_EXPERT], device=device)
        with torch.no_grad():
            out = model(**inputs, dataset_index=dataset_index)
        pose = processor.post_process_pose_estimation(out, boxes=[box])[0][0]
        kp, score = pose["keypoints"].cpu().numpy(), pose["scores"].cpu().numpy()
        for i, ((x, y), s) in enumerate(zip(kp, score)):
            rows.append((args.model_name, rec.frame, rec.camera, names[i], float(x), float(y), float(s)))
    pd.DataFrame(rows, columns=["model", "frame", "camera", "keypoint", "x_px", "y_px", "conf"]).to_csv(args.out, index=False)
    print(args.out, len(rows), "rows; keypoints:", names)


if __name__ == "__main__":
    main()
