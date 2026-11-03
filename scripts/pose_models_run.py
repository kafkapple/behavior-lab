"""Run the web app's own pose inference (SUBTLE_gpu `run_tools_pose_inference`) locally on one video.

Same code path as the server: composite -> camera cells -> model -> normalized keypoint json.gz,
followed by the server's default skeleton3d post-processing. Weights and server code are copies
(see docs: vault note 261103_AVATAR_pose_models_server_vs_local).

    pose_runtime/.venv/bin/python scripts/pose_models_run.py --algorithm sleap \
        --model ~/data/subtle_models/SLEAP_1423/<dir> --video <composite.mp4> --out <result.json.gz> --device mps
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

SERVER_CODE = Path.home() / "data/subtle_outage_poc_261030/SUBTLE_gpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", choices=["sleap", "yolo", "lightning_pose"], required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--config", type=Path, default=None, help="server pose config; default = server default per algorithm")
    ap.add_argument("--map-location", choices=["cpu"], default=None,
                    help="default torch.load map_location (Deviation: CUDA-saved Lightning Pose checkpoints cannot be read on a Mac otherwise)")
    ap.add_argument("--server-code", type=Path, default=SERVER_CODE)
    a = ap.parse_args()
    if a.out.exists():
        ap.error(f"output exists, use a new path: {a.out}")
    for p in (a.model, a.video, a.server_code):
        if not p.exists():
            ap.error(f"missing: {p}")

    # the server writes <video>.json.gz beside its input: give it a private symlink, never the source folder
    work = a.out.parent / f".work_{a.out.name.split('.')[0]}"
    work.mkdir(parents=True, exist_ok=True)
    link = work / a.video.name
    if not link.exists():
        link.symlink_to(a.video.resolve())

    if a.map_location:  # driver-side only: the server code is not edited
        import torch
        _load = torch.load
        torch.load = lambda *args, **kw: _load(*args, **{"map_location": a.map_location, **kw})

    sys.path.insert(0, str(a.server_code))
    import infer_pose_tools as tools  # noqa: E402  (server module, needs the path above)

    model = a.model
    if a.algorithm == "lightning_pose" and model.suffix == ".zip":  # the server unpacks the zip first
        model = tools.extract_lightning_pose_model_archive(model, a.out.parent / f".lp_{model.stem}")

    t0 = time.time()
    info = tools.run_tools_pose_inference(
        a.algorithm, input_video=link, output_path=a.out, device=a.device,
        model_path=model, pose_config_path=a.config,
    )
    d = json.load(gzip.open(a.out))
    frames = len(d["keypoint"])
    assert frames > 0 and d["node"]["cnt"] > 0, "empty result"
    meta = {"algorithm": a.algorithm, "model": str(a.model), "device": a.device, "frames": frames,
            "seconds": round(time.time() - t0, 1), "actual_device": info.get("actual_device"),
            "json_model": d.get("model"), "json_weight": d.get("weight")}
    a.out.with_suffix("").with_suffix(".run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
