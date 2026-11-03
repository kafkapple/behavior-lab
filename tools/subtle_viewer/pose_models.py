"""Model registry and one-click prediction for the Models tab. No web code. Prediction runs pose_session.py in the pose_runtime venv (never in the UI thread)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path

import yaml

import core

REPO = Path(__file__).resolve().parents[2]
CALIB = Path(os.environ.get("SUBTLE_CALIB", Path.home() / "data/subtle_outage_poc_261030/jh_config.toml"))
# YOLO and Lightning Pose run through the server's own pipeline (run_server.py, one runner shared with the pose-comparison notes)
# ponytail: falls back to the main checkout's copy until feat/261029_subtle_tutorial has the commit; drop the fallback after the merge
RUN_SERVER = next((p for p in (REPO / "src/behavior_lab/pose/predictors/run_server.py", Path.home() / "dev/behavior-lab/src/behavior_lab/pose/predictors/run_server.py") if p.is_file()), None)
LP_PY = Path.home() / "data/subtle_models/_envs/lp/bin/python"       # kornia pinned 0.8.3 (0.9.0 breaks Lightning Pose output)
SLEAP11 = ["nose1", "neck1", "earL1", "earR1", "forelegL1", "forelegR1", "tailstart1", "hindlegL1", "hindlegR1", "tail1", "tailend1"]
SERVER_PRESETS = {  # preset -> (label, nodes, runtime, viewer file tag)
    "yolo": ("YOLO11m AVATAR3D (server rule: box centres)", ["nose", "head", "ass", "chest", "rleg", "lleg", "rarm", "larm", "tail"], "pose_runtime", "yolo"),
    "lp_reorganize": ("Lightning Pose 20260717_reorganize (server model)", SLEAP11, "lp", "lpreorg"),
    "lp_khu": ("Lightning Pose KHU (server model)", SLEAP11, "lp", "lpkhu"),
}
JOBS: dict = {}
LOCK = threading.Lock()


def registry(results: Path) -> list[dict]:
    """SLEAP single_instance models (the only kind sleap-nn can run from one folder): ~/data/subtle_models, trained ones under <results>/pose_jobs."""
    roots = [Path(os.environ.get("SUBTLE_MODELS", Path.home() / "data/subtle_models")), results / "pose_jobs", results / "local_pose" / "models"]
    out, seen = [], set()
    for r in roots:
        for ck in sorted(r.rglob("best.ckpt")) if r.is_dir() else []:
            d = ck.parent
            if ck.stat().st_size in seen:      # the same checkpoint copied into another root
                continue
            seen.add(ck.stat().st_size)
            try:
                c = yaml.safe_load((d / "training_config.yaml").read_text())
                heads = [k for k, v in c["model_config"]["head_configs"].items() if v]
                nodes = [n["name"] for n in c["data_config"]["skeletons"][0]["nodes"]]
            except (OSError, KeyError, yaml.YAMLError):
                continue
            out.append({"path": str(d), "name": d.name, "head": heads[0] if heads else "?", "nodes": nodes, "usable": heads == ["single_instance"],
                        "tag": re.search(r"n=(\d+)", d.name).group(1) if re.search(r"n=(\d+)", d.name) else d.name[:12]})
    for name, (label, nodes, rt, _) in SERVER_PRESETS.items():
        py = LP_PY if rt == "lp" else REPO / "tools/pose_runtime/.venv/bin/python"
        out.append({"path": f"preset:{name}", "name": label, "head": "server pipeline, needs the composite video", "nodes": nodes, "usable": RUN_SERVER is not None and py.is_file(), "tag": SERVER_PRESETS[name][3]})
    return out


def _start_server(root: Path, results: Path, sessions: list[dict], body: dict) -> dict:
    """YOLO / Lightning Pose: run_server.py on the composite video, then store the server json next to the source as <stem>_sleap_<tag>.json.gz
    (the `_sleap_` part is only the viewer's pairing convention for camera videos)."""
    name = body["model"].split(":", 1)[1]
    if name not in SERVER_PRESETS or RUN_SERVER is None:
        raise ValueError("unknown preset or run_server.py missing")
    s = next((x for x in sessions if x["path"] == body["session"]), None)
    vid = body.get("video") or (s or {}).get("video")
    if not vid or not (Path(vid) if Path(vid).is_absolute() else root / vid).is_file():
        raise ValueError("this model needs the composite video (one mp4 with the five camera cells): give its path")
    vid = Path(vid) if Path(vid).is_absolute() else root / vid
    _, _, rt, tag = SERVER_PRESETS[name]
    python = LP_PY if rt == "lp" else REPO / "tools/pose_runtime/.venv/bin/python"
    if not python.is_file():
        raise ValueError(f"missing runtime: {python}")
    with LOCK:
        if any(j["state"] == "running" for j in JOBS.values()):
            raise ValueError("another prediction is running")
        jid = uuid.uuid4().hex[:10]
        out = results / "pose_sessions" / jid
        out.mkdir(parents=True, exist_ok=True)
        job = {"id": jid, "state": "running", "log": "", "model": SERVER_PRESETS[name][0]}
        JOBS[jid] = job
        p = subprocess.Popen([str(python), str(RUN_SERVER), "--preset", name, "--video", str(vid), "--out", str(out / "result.json.gz")],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        job["process"] = p
    src = root / body["session"]

    def work():
        job["log"] = p.stdout.read()
        code = p.wait()
        with LOCK:
            res = out / "result.json.gz"
            if code == 0 and res.is_file():
                dst = src.with_name(src.name.removesuffix(".json.gz").split("_sleap_")[0] + f"_sleap_{tag}.json.gz")
                dst.write_bytes(res.read_bytes())
                job["result"] = {"session": str(dst.relative_to(root)), "source": str(res)}
            else:
                job["result"] = {"error": job["log"][-600:] or f"exit {code}"}
            job["state"] = "done" if "error" not in job["result"] else "failed"
    threading.Thread(target=work, daemon=True).start()
    return {"job": jid}


def start(root: Path, results: Path, sessions: list[dict], body: dict) -> dict:
    """sessions = core.scan items; body = {session, model, video?}. model = a SLEAP folder, or preset:<name> for YOLO / Lightning Pose."""
    if str(body.get("model", "")).startswith("preset:"):
        return _start_server(root, results, sessions, body)
    s = next((x for x in sessions if x["path"] == body["session"]), None)
    if not s or len(s["cams"]) != 5:
        raise ValueError("the open session needs its five camera videos (<stem>_cam_1..5.mp4 in the same folder)")
    m = next((x for x in registry(results) if x["path"] == body["model"]), None)
    if not m or not m["usable"]:
        raise ValueError("choose a SLEAP single_instance model")
    python = REPO / "tools/pose_runtime/.venv/bin/python"
    if not python.is_file():
        raise ValueError("Run uv sync --project tools/pose_runtime first")
    with LOCK:
        if any(j["state"] == "running" for j in JOBS.values()):
            raise ValueError("another prediction is running")
        jid = uuid.uuid4().hex[:10]
        out = results / "pose_sessions" / jid
        out.mkdir(parents=True, exist_ok=True)
        req = {"root": str(root), "session": s["path"], "cams": {str(k): str((root / v).resolve()) for k, v in s["cams"].items()}, "model": m["path"], "tag": m["tag"], "calib": str(CALIB), "out": str(out)}
        job = {"id": jid, "state": "running", "log": "", "model": m["name"]}
        JOBS[jid] = job
        p = subprocess.Popen([str(python), str(REPO / "tools/subtle_viewer/pose_session.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        job["process"] = p

    def work():
        p.stdin.write(json.dumps(req)); p.stdin.close()
        threading.Thread(target=lambda: [job.__setitem__("log", job["log"] + l) for l in p.stderr], daemon=True).start()
        raw, code = p.stdout.read(), p.wait()
        with LOCK:
            try:
                job["result"] = json.loads(raw.strip().splitlines()[-1])
            except (ValueError, IndexError):
                job["result"] = {"error": job["log"][-600:] or f"exit {code}"}
            job["state"] = "done" if code == 0 and "error" not in job["result"] else "failed"
    threading.Thread(target=work, daemon=True).start()
    return {"job": jid}


def status(jid: str) -> dict:
    j = JOBS[jid]
    return {k: (v[-2000:] if k == "log" else v) for k, v in j.items() if k != "process"}
