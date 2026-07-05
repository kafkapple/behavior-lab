"""VAME wrapper: RNN-VAE motion embedding + latent motif clustering.

Reference: Luxem et al. (2022), Communications Biology.
Maintained fork: EthoML/VAME. Install: pip install vame-py  (import name: vame)

VAME is a project/file-based pipeline (no array-in entry point), so this wrapper
shims canonical ``(T,K,D)`` keypoints into a DeepLabCut-format CSV, runs the
standard sequence, and loads the per-frame motif labels + latent vectors back.

API sequence (verified against EthoML/VAME 0.14.x):
    init_new_project -> preprocessing -> create_trainset -> train_model
    -> evaluate_model -> segment_session ; outputs land under results/.
"""
import csv
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from ...core.types import ClusteringResult


class VAME:
    """Thin wrapper around EthoML VAME for behavior-lab ``(T,K,D)`` format.

    Only ``fit_predict`` is required by the discovery seam; the other methods
    satisfy the ``BehaviorClusterer`` protocol.

    Usage:
        model = VAME(project_dir='./vame_output', n_clusters=15)
        result = model.fit_predict(keypoints)   # (T, K, D) -> ClusteringResult
    """

    def __init__(self, project_dir: str = './vame_output',
                 project_name: str = 'behavior_lab_vame',
                 latent_dim: int = 30, time_window: int = 30,
                 n_clusters: int = 15, num_epochs: int = 100,
                 fps: float = 30.0,
                 bodypart_names: Optional[List[str]] = None,
                 centered_keypoint: Optional[str] = None,
                 orientation_keypoint: Optional[str] = None,
                 segmentation_algorithm: str = 'kmeans'):
        self.project_dir = project_dir
        self.project_name = project_name
        self.latent_dim = latent_dim
        self.time_window = time_window
        self.n_clusters = n_clusters
        self.num_epochs = num_epochs
        self.fps = fps
        self.bodypart_names = bodypart_names
        self.centered_keypoint = centered_keypoint
        self.orientation_keypoint = orientation_keypoint
        self.segmentation_algorithm = segmentation_algorithm
        self._config = None
        self._session = 'rec_0'
        self._results_dir = None

    def _write_dlc_csv(self, keypoints: np.ndarray, csv_path: Path,
                       bodyparts: List[str]) -> None:
        """Shim ``(T,K,D)`` -> DeepLabCut multi-index CSV (scorer/bodyparts/coords).

        Writes x,y(+likelihood). For D==3 the first two axes are used (VAME's
        egocentric preprocessing is 2D); z is dropped.
        ponytail: 3D-native VAME is unverified — 2D projection is the safe path.
        """
        T, K, D = keypoints.shape
        xy = keypoints[..., :2]  # (T,K,2)
        scorer = 'behavior_lab'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['scorer'] + [scorer] * (K * 3))
            bp_row = ['bodyparts']
            coord_row = ['coords']
            for bp in bodyparts:
                bp_row += [bp, bp, bp]
                coord_row += ['x', 'y', 'likelihood']
            w.writerow(bp_row)
            w.writerow(coord_row)
            for t in range(T):
                row = [t]
                for k in range(K):
                    row += [float(xy[t, k, 0]), float(xy[t, k, 1]), 1.0]
                w.writerow(row)

    def fit(self, keypoints: np.ndarray) -> 'VAME':
        """Run the full VAME pipeline through segmentation on ``(T,K,D)``."""
        try:
            import vame
        except ImportError:
            raise ImportError(
                "Install VAME: pip install vame-py "
                "(EthoML fork; do NOT `pip install vame`)")

        keypoints = np.asarray(keypoints, dtype=np.float32)
        K = keypoints.shape[1]
        bodyparts = self.bodypart_names or [f'kp{i}' for i in range(K)]
        centered = self.centered_keypoint or self._infer_center(bodyparts)
        orient = self.orientation_keypoint or self._infer_orient(bodyparts)

        proj_root = Path(self.project_dir)
        proj_root.mkdir(parents=True, exist_ok=True)
        csv_path = proj_root / f'{self._session}.csv'
        self._write_dlc_csv(keypoints, csv_path, bodyparts)

        config_file, config = vame.init_new_project(
            project_name=self.project_name,
            poses_estimations=[str(csv_path)],
            working_directory=str(proj_root),
            source_software="DeepLabCut",
            fps=int(round(self.fps)),
        )
        # Push our hyperparameters onto the generated config where supported.
        for key, val in (("zdims", self.latent_dim), ("time_window", self.time_window),
                         ("n_clusters", self.n_clusters), ("max_epochs", self.num_epochs)):
            try:
                config[key] = val
            except Exception:
                pass

        vame.preprocessing(
            config=config,
            centered_reference_keypoint=centered,
            orientation_reference_keypoint=orient,
        )
        vame.create_trainset(config=config)
        vame.train_model(config=config)
        vame.evaluate_model(config=config)
        vame.segment_session(config=config)

        self._config = config
        self._results_dir = Path(config["project_path"]) / "results"
        return self

    def _find_result(self, suffix: str) -> Optional[Path]:
        if self._results_dir is None:
            return None
        hits = sorted(self._results_dir.rglob(suffix))
        return hits[0] if hits else None

    def predict(self, keypoints: np.ndarray) -> np.ndarray:
        """Return per-frame motif labels from the fitted session."""
        if self._results_dir is None:
            raise RuntimeError("Call .fit() first")
        label_file = self._find_result(f"*_label_{self._session}.npy") \
            or self._find_result("*label*.npy")
        if label_file is None:
            raise RuntimeError("VAME label output not found under results/")
        return np.load(label_file)

    def get_embeddings(self, keypoints: np.ndarray) -> np.ndarray:
        """Return per-frame latent vectors (T, latent_dim)."""
        if self._results_dir is None:
            raise RuntimeError("Call .fit() first")
        emb_file = self._find_result("latent_vectors.npy")
        if emb_file is None:
            raise RuntimeError("VAME latent_vectors.npy not found under results/")
        return np.load(emb_file)

    def fit_predict(self, keypoints: np.ndarray) -> ClusteringResult:
        """Fit and return a structured ClusteringResult."""
        self.fit(keypoints)
        labels = np.asarray(self.predict(keypoints))
        try:
            emb = np.asarray(self.get_embeddings(keypoints))
        except RuntimeError:
            emb = None
        return ClusteringResult(
            labels=labels,
            embeddings=emb[:, :2] if emb is not None and emb.shape[1] >= 2 else emb,
            n_clusters=len(set(labels.tolist())),
            features=emb,
            metadata={"algorithm": "vame", "latent_dim": self.latent_dim,
                      "segmentation": self.segmentation_algorithm},
        )

    def _infer_center(self, bodyparts: List[str]) -> str:
        for name in ("spine", "center", "body", "torso", "back"):
            if name in bodyparts:
                return name
        return bodyparts[len(bodyparts) // 2]

    def _infer_orient(self, bodyparts: List[str]) -> str:
        for name in ("tail_base", "tail", "tailbase", "rump", "nose", "snout"):
            if name in bodyparts:
                return name
        return bodyparts[-1]

    def save(self, path: str) -> None:
        state = {
            "config": self._config,
            "results_dir": str(self._results_dir) if self._results_dir else None,
            "session": self._session,
            "latent_dim": self.latent_dim,
            "n_clusters": self.n_clusters,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._config = state.get("config")
        rd = state.get("results_dir")
        self._results_dir = Path(rd) if rd else None
        self._session = state.get("session", self._session)
        self.latent_dim = state.get("latent_dim", self.latent_dim)
        self.n_clusters = state.get("n_clusters", self.n_clusters)
