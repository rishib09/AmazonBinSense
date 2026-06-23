"""
Environment detection and path resolution for BinSense notebooks.

Code and data can now live in different places (code = git repo on D:/GitHub_Repo,
data = Google Drive). Set BINSENSE_DATA_DIR to the data root to override.

Usage at the top of every notebook:
    from utils.env_utils import setup_env
    cfg = setup_env()
    # then use cfg.base_dir (code), cfg.data_dir / cfg.images_dir (data), etc.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Environment detection ─────────────────────────────────────────────────────


def _detect_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


IN_COLAB: bool = _detect_colab()


# ── Path configuration ────────────────────────────────────────────────────────


@dataclass
class BinSensePaths:
    base_dir: Path   # code root (git repo)
    data_dir: Path   # data root (may differ from base_dir — set via BINSENSE_DATA_DIR)
    images_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    notebooks_dir: Path = field(init=False)
    labels_dir: Path = field(init=False)
    splits_dir: Path = field(init=False)
    mlflow_dir: Path = field(init=False)

    def __post_init__(self):
        self.images_dir   = self.data_dir / "images"
        self.metadata_dir = self.data_dir / "metadata"
        self.labels_dir   = self.data_dir / "labels"
        self.splits_dir   = self.data_dir / "splits"
        self.models_dir   = self.base_dir / "models"
        self.notebooks_dir = self.base_dir / "notebooks"
        self.mlflow_dir   = self.base_dir / "mlruns"

    def makedirs(self) -> None:
        for p in [
            self.data_dir,
            self.images_dir,
            self.metadata_dir,
            self.models_dir,
            self.labels_dir,
            self.splits_dir,
            self.mlflow_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


def _mount_drive_if_needed() -> None:
    if not IN_COLAB:
        return
    from google.colab import drive

    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def _resolve_base_dir() -> Path:
    """
    Code root resolution order:
      1. BINSENSE_DIR env var (Docker / CI override)
      2. __file__-relative: utils/env_utils.py -> parent.parent == project root
    """
    env_override = os.getenv("BINSENSE_DIR")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parents[1]


def _resolve_data_dir(base_dir: Path) -> Path:
    """
    Data root resolution order:
      1. BINSENSE_DATA_DIR env var — set in notebook bootstrap when code and
         data live in different places (e.g. code = git clone, data = Drive)
      2. base_dir / "data" — default (single-directory layout, local dev)
    """
    env_override = os.getenv("BINSENSE_DATA_DIR")
    if env_override:
        return Path(env_override)
    return base_dir / "data"


# ── Main entry point ──────────────────────────────────────────────────────────


def setup_env(verbose: bool = True) -> BinSensePaths:
    """
    Mount Drive (Colab only), resolve paths, create directories, return cfg.

    Example:
        cfg = setup_env()
        df = pd.read_csv(cfg.splits_dir / "seed.csv")
        img = cfg.images_dir / "00015.jpg"
    """
    _mount_drive_if_needed()
    base_dir = _resolve_base_dir()
    data_dir = _resolve_data_dir(base_dir)
    paths = BinSensePaths(base_dir=base_dir, data_dir=data_dir)
    paths.makedirs()

    if verbose:
        env_label = "Google Colab" if IN_COLAB else "Local (VS Code)"
        print(f"[BinSense] Running in: {env_label}")
        print(f"[BinSense] Code dir  : {paths.base_dir}")
        print(f"[BinSense] Data dir  : {paths.data_dir}")

    return paths


# ── GPU helpers ───────────────────────────────────────────────────────────────


def get_device() -> "torch.device":
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


def print_gpu_info() -> None:
    import torch

    device = get_device()
    print(f"[BinSense] Device: {device}")
    if device.type == "cuda":
        print(f"           GPU   : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"           VRAM  : {mem:.1f} GB")
