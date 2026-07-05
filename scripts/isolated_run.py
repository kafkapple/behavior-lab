"""Run ONE discovery method inside an isolated conda env, save labels/features.

Heavy tools (keypoint-moseq/jax, VAME) conflict with the main torch env's numpy,
so each lives in its own conda env. This standalone runner uses behavior_lab via
PYTHONPATH (numpy/scipy/sklearn only — no torch), runs the method, and writes
``labels.npy`` (+ ``features.npy``) for the MAIN env to score uniformly.

Usage (from the repo root):
    conda run -n kpms python scripts/isolated_run.py \
        --method keypoint_moseq --npz <file.npz> --out outputs/iso/kpms
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from behavior_lab.data import ingest  # noqa: E402
from behavior_lab.models import get_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="e.g. keypoint_moseq, vame, subtle")
    ap.add_argument("--npz", required=True, help="keypoint file to ingest")
    ap.add_argument("--out", required=True, help="output dir for labels/features")
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--n-clusters", type=int, default=15)
    a = ap.parse_args()

    seq = ingest(a.npz)[0]
    kp = seq.keypoints[: a.max_frames]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.method in ("keypoint_moseq", "moseq"):
        model = get_model(
            a.method,
            project_dir=str(out / "proj"),
            num_iters=10,
            latent_dim=6,
            bodypart_names=[f"kp{i}" for i in range(kp.shape[1])],
        )
        result = model.fit_predict(kp)
    elif a.method == "vame":
        result = get_model("vame", project_dir=str(out / "proj"),
                           n_clusters=a.n_clusters, num_epochs=20, fps=seq.fps).fit_predict(kp)
    elif a.method == "subtle":
        result = get_model("subtle", fps=int(round(seq.fps))).fit_predict([kp], isolate=True)
    else:
        result = get_model(a.method).fit_predict(kp)

    np.save(out / "labels.npy", np.asarray(result.labels))
    if result.features is not None:
        np.save(out / "features.npy", np.asarray(result.features))
    print(f"ISOLATED_DONE method={a.method} n_clusters={result.n_clusters} "
          f"len={len(result.labels)} features={'yes' if result.features is not None else 'no'}")


if __name__ == "__main__":
    main()
