"""Make small per-camera proxy videos for the project sessions, for overlay viewing only (inference must use the raw video).
One session at a time: copy the 5-camera composite from the server, cut the five camera cells in ONE decoding pass (cells from core.CELLS),
write half-size H.264 proxies next to the session file, check the frame count against the session, delete the copy. Safe to re-run.
usage: fetch_cams.py <root> --project NAME [--host eren] [--remote-dir DIR] [--only veh2,cort3] [--keep-raw]"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import core

log = logging.getLogger("fetch_cams")
FF, FP = "/opt/homebrew/bin/ffmpeg", "/opt/homebrew/bin/ffprobe"
REMOTE = "/home/data/kafkapple/avatar_gslrm/raw_full/20231205_CORT_drinking_28d"      # composites: <animal>.mp4
PREFIX = "mouse_avatar_231205_cort28d_"                                                # proxy name = PREFIX + animal + _cam_N.mp4 (scan pairs by animal token and cam_N)
MIN_FREE_GB = 14                                                                        # one composite is about 10 GB


def animal_of(rel: str) -> str:
    m = re.match(r"([A-Za-z]+\d+)", Path(rel).name)
    if not m:
        raise ValueError(f"no animal token in {rel}")
    return m.group(1)


def frames(path: Path) -> int:
    return int(subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                              capture_output=True, text=True, check=True).stdout.strip())


def cut_command(src: Path, outs: dict[int, Path]) -> list[str]:
    """One ffmpeg pass: split the decoded composite and crop + halve each camera cell. Same encoder settings as make_proxy.sh."""
    cams = sorted(outs)
    split = f"[0:v]split={len(cams)}" + "".join(f"[s{c}]" for c in cams)
    crops = ";".join(f"[s{c}]crop={core.CELLS[c][2]}:{core.CELLS[c][3]}:{core.CELLS[c][0]}:{core.CELLS[c][1]},scale={core.CELLS[c][2] // 4 * 2}:{core.CELLS[c][3] // 4 * 2}[o{c}]" for c in cams)
    cmd = [FF, "-v", "error", "-y", "-i", str(src), "-filter_complex", f"{split};{crops}"]
    for c in cams:
        cmd += ["-map", f"[o{c}]", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", "-g", "20", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(outs[c])]
    return cmd


def one(root: Path, rel: str, host: str, remote: str, keep_raw: bool) -> str:
    animal, folder = animal_of(rel), core.safe(root, rel).parent
    outs = {c: folder / f"{PREFIX}{animal}_cam_{c + 1}.mp4" for c in core.CELLS}
    if all(p.exists() and p.stat().st_size > 0 for p in outs.values()):
        return "skip (all five proxies exist)"
    n_expected = len(core.load(root, rel)["xyz"])
    if shutil.disk_usage(root).free / 1e9 < MIN_FREE_GB:
        raise RuntimeError(f"less than {MIN_FREE_GB} GB free")
    raw = root / ".raw_copy" / f"{animal}.mp4"
    raw.parent.mkdir(exist_ok=True)
    log.info("%s: copying composite from %s", animal, host)
    subprocess.run(["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=5", f"{host}:{remote}/{animal}.mp4", str(raw)], check=True)
    tmp = {c: p.with_suffix(".tmp.mp4") for c, p in outs.items()}
    log.info("%s: cutting five cameras in one pass", animal)
    subprocess.run(cut_command(raw, tmp), check=True)
    bad = {c: frames(p) for c, p in tmp.items() if frames(p) != n_expected}
    if bad:
        for p in tmp.values(): p.unlink(missing_ok=True)
        raise RuntimeError(f"{animal}: frame count differs from the session ({n_expected}): {bad}")
    for c, p in tmp.items():
        p.replace(outs[c])
    if not keep_raw:
        raw.unlink()
    return f"ok {n_expected} frames, " + ", ".join(f"cam_{c + 1} {outs[c].stat().st_size / 1e6:.1f} MB" for c in sorted(outs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path); ap.add_argument("--project", required=True); ap.add_argument("--host", default="eren")
    ap.add_argument("--remote-dir", default=REMOTE); ap.add_argument("--only", default=""); ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    root = a.root.resolve()
    paths = [r for s in core.load_project(root, a.project)["groups"].values() for r in s]
    only = set(filter(None, a.only.split(",")))
    failed = 0
    for rel in paths:
        if only and animal_of(rel) not in only:
            continue
        try:
            log.info("%s -> %s", rel, one(root, rel, a.host, a.remote_dir, a.keep_raw))
        except Exception as e:  # keep going: one bad animal must not block the rest
            failed += 1; log.error("%s FAILED: %s", rel, e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
