# AVATAR rig — code entry point

> Rig facts (server paths, calibration, layout, formats by date, residual by date) are owned by the vault
> manual `30_Projects/2609_IBS_Postdoc/_Manual/Lab-Server/260928_AVATAR_Multiview_데이터셋_매뉴얼.md`.
> This page only says where the code is and how to run it. Do not copy numbers here.

## Where the code lives (261002)

| Step | Owner | Path |
|---|---|---|
| Rig layout (composite size, cell origins, calibration date) | behavior-lab | `configs/rig/avatar.yaml` |
| Composite → per-camera mp4 | behavior-lab | `behavior_lab.rig.avatar.split_composite` |
| Calibration loading, DLT, reprojection residual | behavior-lab | `behavior_lab.rig.multiview` |
| Calibration residual by recording date, plot | behavior-lab | `behavior_lab.rig.avatar` (`residual`, `plot`), figure code in `behavior_lab.visualization.rig` |
| 2D keypoints (SUBTLE SLEAP) | outside any repo | Olaf `~/data/avatar_gslrm/eren_avatar_subtle_260930/run_sleap_one.py` |
| Masks (SAM3 tracker) | BehaviorSplatter | `scripts/data_prep/avatar_sam3_tracker_masks.py` |
| 3D keypoints in GS-LRM space | BehaviorSplatter | `scripts/data_prep/avatar_keypoints_to_gslrm.py` |
| Reader, undistortion, 512² crop | BehaviorSplatter | `src/behaviorsplatter/preprocessing/_reader_avatar.py`, `configs/preprocessing/avatar_*.yaml` |

Rule: code that only needs the rig (videos, `config.toml`, 2D keypoints) lives here; code that needs
the GS-LRM preprocessing or training stack stays in BehaviorSplatter.

## Commands

```bash
python -m behavior_lab.rig.avatar split --composite rec.mp4 --calib config.toml --out <dir> --prefix <name>
python -m behavior_lab.rig.avatar residual --calib config.toml --kp TAG=YYYYMMDD=<json.gz> ... --out drift.json
python -m behavior_lab.rig.avatar plot --json drift.json --out drift.png
pytest tests/test_data/test_rig_avatar.py
```

Needs the `viz` extra (opencv, matplotlib) and `ffmpeg` / `ffprobe` on PATH for `split`.

## Bottom camera before 2023-08-22

In recordings up to 2023-05-23 the bottom camera image is a quarter turn away from the calibrated one.
`behavior_lab.rig.multiview.rotate_image_90` derives the matching camera table (no fitting) and
`write_calib` writes it. The derived file used for experiments is
`~/data/avatar_gslrm/config_cam3rot90_until_20230523.toml` (Mac and Olaf); the original `config.toml`
is unchanged and stays the file for recordings from 2023-08-22 on. Numbers: vault note
`261002_AVATAR_calib_residual_by_date.md`.

## Open

- BehaviorSplatter keeps `load_calib` / `triangulate` / `reproj_px` inside
  `scripts/data_prep/avatar_keypoints_to_gslrm.py`: five analysis scripts import them and the
  weighted `triangulate` exists only there. Its split and residual-by-date scripts were removed
  (BS dev `81171ffd`). Revisit only if behavior-lab becomes a BS dependency.
- `~/dev/behavior-tools`: the earlier POC is parked on branch `archive/260928_avatar_rig_poc`
  (local only), not on its main.
- Not covered: recordings that are not 3600x2000, per-date extrinsic re-estimation for recordings
  before 2023-08-22.
- `run_sleap_one.py` stays outside the repos on purpose: it is an 11-line caller of the SUBTLE
  checkout next to it (`run_tools_pose_inference("sleap", ...)` with
  `configs/SLEAP/AVATAR3D_11_config.json` and `model/SLEAP/260504.single_instance.n=1423-1`) and
  cannot run without that checkout.
