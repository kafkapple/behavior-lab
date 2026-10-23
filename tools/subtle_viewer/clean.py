"""List (and optionally delete) old result folders. Keeps the latest KEEP_LATEST folders per project and any folder that contains a file named KEEP.
Default is a dry run; --apply deletes the listed folders after printing them. usage: clean.py [--results-dir DIR] [--apply]"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KEEP_LATEST = 3


def stale(results_base: Path) -> list[Path]:
    """Result folders beyond retention, oldest first. Folder names start with YYMMDD_HHMMSS, so name order is time order."""
    out = []
    for proj in sorted(p for p in results_base.glob("*") if p.is_dir()):
        runs = sorted(p for p in proj.iterdir() if p.is_dir())
        out += [r for r in runs[: max(0, len(runs) - KEEP_LATEST)] if not (r / "KEEP").exists()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "subtle")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    old = stale(a.results_dir)
    for p in old:
        print(("delete " if a.apply else "would delete ") + str(p))
    if a.apply:
        for p in old:
            shutil.rmtree(p)
    print(f"{len(old)} folder(s) beyond the latest {KEEP_LATEST} per project" + ("" if a.apply else " (dry run; --apply deletes)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
