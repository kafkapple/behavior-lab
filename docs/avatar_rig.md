# AVATAR rig — code entry point

> Rig facts (server paths, calibration, layout, formats by date, residual by date) are owned by the vault
> manual `30_Projects/2609_IBS_Postdoc/_Manual/Lab-Server/260928_AVATAR_Multiview_데이터셋_매뉴얼.md`.
> This page only says where the code is and how to run it. Do not copy numbers here.

## Where the code lives (261002)

| Step | Owner | Path |
|---|---|---|
| Composite → per-camera mp4 | behavior-lab | `behavior_lab.rig.avatar.split_composite` |
| Calibration loading, DLT, reprojection residual | behavior-lab | `behavior_lab.rig.multiview` |
| Calibration residual by recording date, plot | behavior-lab | `behavior_lab.rig.avatar` (`residual`, `plot`) |
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

## Open (duplicates to retire)

- BehaviorSplatter still carries the originals: `scripts/data_prep/avatar_split_composite.py`, the
  `load_calib` / `triangulate` / `reproj_px` functions inside `avatar_keypoints_to_gslrm.py`, and
  `scripts/analysis/_exp_261002_avatar_calib_drift_{by_date,plot}.py`. They were left untouched on
  261002 because training jobs were reading that checkout. This module reproduces the BS residuals
  on the 12 dated samples to 1e-12 px.
- `~/dev/behavior-tools` has an earlier POC (`splitter`, uncommitted `calibration/`). Superseded by
  this module; its 1200x1000-for-every-cell default cuts the bottom camera.
- Not covered: recordings that are not 3600x2000, per-date extrinsic re-estimation for recordings
  before 2023-08-22.
