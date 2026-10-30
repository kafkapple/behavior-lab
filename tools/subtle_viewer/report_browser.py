"""Browse saved local analysis artifacts for projects in the current data root."""
from __future__ import annotations

import re
from pathlib import Path

import core

FILE_TYPES = {".md", ".png", ".csv", ".json", ".mp4"}


def _part(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError("invalid result path")
    return value


def artifact(root: Path, project: str, run: str, file: str) -> Path:
    if project not in core.list_projects(root):
        raise ValueError("project is not in this data root")
    base = core.results_dir(root, _part(project)).resolve()
    path = (base / _part(run) / _part(file)).resolve()
    if base not in path.parents or not path.is_file() or path.suffix.lower() not in FILE_TYPES:
        raise ValueError("unknown result file")
    return path


def list_reports(root: Path, project: str) -> list[dict]:
    if project not in core.list_projects(root):
        raise ValueError("project is not in this data root")
    base = core.results_dir(root, _part(project))
    if not base.is_dir():
        return []
    reports = []
    for run in sorted(base.iterdir(), key=lambda p: p.name, reverse=True):
        if not run.is_dir() or run.is_symlink() or not re.fullmatch(r"[A-Za-z0-9_.-]+", run.name):
            continue
        files = [p.name for p in sorted(run.iterdir()) if p.is_file() and p.suffix.lower() in FILE_TYPES and not p.is_symlink()]
        if files:
            readme = run / "README.md"
            reports.append({"run": run.name, "files": files, "readme": readme.read_text(encoding="utf-8") if readme.is_file() and not readme.is_symlink() else ""})
    return reports
