"""Dataset loader for SLEAP pose exports."""
from __future__ import annotations

from pathlib import Path

from ...core.types import BehaviorSequence
from ...pose.sleap import load_sleap_file


class SLEAPLoader:
    """Load SLEAP ``.slp``/analysis ``.h5`` files as BehaviorSequence objects."""

    def __init__(
        self,
        data_dir: str | Path,
        skeleton_name: str = "sleap",
        fps: float = 30.0,
        instance_mode: str = "flatten",
        confidence_threshold: float | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.instance_mode = instance_mode
        self.confidence_threshold = confidence_threshold

    def load_file(self, filepath: str | Path) -> list[BehaviorSequence]:
        result = load_sleap_file(
            filepath,
            fps=self.fps,
            skeleton_name=self.skeleton_name,
            instance_mode=self.instance_mode,
            confidence_threshold=self.confidence_threshold,
        )
        return result.sequences

    def load_all(self) -> list[BehaviorSequence]:
        sequences: list[BehaviorSequence] = []
        for pattern in ("*.slp", "*.h5", "*.hdf5"):
            for path in sorted(self.data_dir.glob(pattern)):
                sequences.extend(self.load_file(path))
        if not sequences:
            raise FileNotFoundError(f"No SLEAP .slp/.h5/.hdf5 files found in {self.data_dir}")
        return sequences

    def load_split(self, split: str = "train") -> list[BehaviorSequence]:
        split_dir = self.data_dir / split
        if split_dir.exists():
            return SLEAPLoader(
                split_dir,
                skeleton_name=self.skeleton_name,
                fps=self.fps,
                instance_mode=self.instance_mode,
                confidence_threshold=self.confidence_threshold,
            ).load_all()
        return self.load_all()


__all__ = ["SLEAPLoader"]
