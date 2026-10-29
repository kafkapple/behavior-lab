"""Small, real video-to-keypoint AVATAR tutorial using the verified eren checkpoint."""
from __future__ import annotations

import base64
import gzip
import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2

import core

REMOTE = Path("/home/data/kafkapple/avatar_subtle_261004/SUBTLE_gpu")
SOURCE = Path("/home/data/kafkapple/avatar_gslrm/raw_full/20231205_CORT_drinking_28d/cort1.mp4")
FRAMES = 300
JOBS: dict[str, dict] = {}
log = logging.getLogger("subtle_viewer.tutorial")


def _run(cmd: list[str], *, cwd: Path | None = None):
    result = subprocess.run(cmd, cwd=cwd, timeout=1800, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-700:] or f"command exited {result.returncode}")


def _clip_script() -> str:
    return """import cv2,sys
src,dst,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
cap=cv2.VideoCapture(src)
if not cap.isOpened(): raise RuntimeError('source video could not open')
w,h=int(cap.get(3)),int(cap.get(4)); fps=cap.get(5) or 20
out=cv2.VideoWriter(dst,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
if not out.isOpened(): raise RuntimeError('toy clip could not open for writing')
count=0
while count<n:
 ok,frame=cap.read()
 if not ok: break
 out.write(frame); count+=1
out.release();cap.release()
if count!=n: raise RuntimeError(f'expected {n} frames, got {count}')
"""


def _remote(args: list[str], *, cwd: Path | None = None):
    command = (f"cd {shlex.quote(str(cwd))} && " if cwd else "") + shlex.join(args)
    _run(["ssh", "eren", command])


def _split_cameras(video: Path, dest: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("downloaded tutorial video could not open")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width, height = int(cap.get(3)), int(cap.get(4))
    writers = []
    for cam, (x, y, w, h) in core.CELLS.items():
        if x + w > width or y + h > height:
            raise RuntimeError(f"camera {cam + 1} cell exceeds video size {width}x{height}")
        p = dest / f"cort1_toy_cam_{cam + 1}.mp4"
        writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"could not write {p}")
        writers.append((writer, x, y, w, h))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for writer, x, y, w, h in writers:
            writer.write(frame[y:y + h, x:x + w])
        count += 1
    cap.release()
    for writer, *_ in writers:
        writer.release()
    return count


def start(root: Path) -> str:
    if any(not j["done"] for j in JOBS.values()):
        raise ValueError("tutorial inference is already running")
    job_id = uuid.uuid4().hex[:10]
    job = JOBS[job_id] = {"stage": "queued", "pct": 0, "done": False, "error": None, "t0": time.time()}
    threading.Thread(target=_execute, args=(root, job), daemon=True).start()
    return job_id


def _execute(root: Path, job: dict):
    remote_tmp = f"/tmp/subtle_tutorial_{uuid.uuid4().hex[:10]}"
    dest = root / "tutorial"
    local_tmp = dest / (".build_" + uuid.uuid4().hex[:8])
    try:
        dest.mkdir(parents=True, exist_ok=True)
        local_tmp.mkdir()
        job.update(stage=f"Cutting {FRAMES} real video frames on eren", pct=10)
        _remote(["mkdir", "-p", f"{remote_tmp}/result"])
        script = base64.b64encode(_clip_script().encode()).decode()
        code = f"exec(__import__('base64').b64decode('{script}'))"
        _remote([str(REMOTE.parent / ".venv/bin/python"), "-c", code, str(SOURCE), f"{remote_tmp}/cort1_toy.mp4", str(FRAMES)])
        job.update(stage="Predicting 2D keypoints and 3D skeleton on eren GPU", pct=30)
        _remote([str(REMOTE.parent / ".venv/bin/python"), str(REMOTE.parent / "_exp_261004_avatar_sleap_eren.py"),
                 f"{remote_tmp}/cort1_toy.mp4", f"{remote_tmp}/result/cort1_toy.json.gz"], cwd=REMOTE)
        job.update(stage="Downloading toy video and predictions to this Mac", pct=70)
        _run(["scp", f"eren:{remote_tmp}/cort1_toy.mp4", f"eren:{remote_tmp}/result/cort1_toy.json.gz", str(local_tmp)])
        with gzip.open(local_tmp / "cort1_toy.json.gz", "rt") as f:
            data = json.load(f)
        if len(data.get("skeleton3d", {})) != FRAMES or not data.get("keypoint"):
            raise RuntimeError(f"inference result lacks the expected {FRAMES}-frame skeleton or keypoints")
        job.update(stage="Splitting five camera views and checking output", pct=85)
        count = _split_cameras(local_tmp / "cort1_toy.mp4", local_tmp)
        if count != FRAMES:
            raise RuntimeError(f"downloaded video has {count} frames, expected {FRAMES}")
        provenance = {"source_video": str(SOURCE), "host": "eren", "frames": FRAMES, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "inference": str(REMOTE.parent / "_exp_261004_avatar_sleap_eren.py"),
                      "keypoint_model": str(REMOTE / "model/SLEAP/260504.single_instance.n=1423-1/best.ckpt"),
                      "config": str(REMOTE / "configs/SLEAP/AVATAR3D_11_config.json")}
        (local_tmp / "provenance.json").write_text(json.dumps(provenance, indent=2))
        for p in local_tmp.iterdir():
            os.replace(p, dest / p.name)
        job.update(stage="Ready: tutorial/cort1_toy.json.gz", pct=100, done=True, path="tutorial/cort1_toy.json.gz")
    except Exception as exc:
        log.exception("tutorial inference failed")
        job.update(stage="failed", done=True, error=f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(local_tmp, ignore_errors=True)
        try:
            _remote(["rm", "-rf", remote_tmp])
        except Exception:
            log.warning("remote tutorial temporary folder remains: %s", remote_tmp)


def status(job_id: str) -> dict:
    job = JOBS[job_id]
    return {**job, "elapsed": round(time.time() - job["t0"], 1)}
