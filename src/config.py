"""
Central configuration: file paths, random seeds, hyper-parameters and business costs.

Why a single config module?
---------------------------
Every "magic number" in the project (seed 42, the 60/20/20 split, the MLP
hyper-parameters, the ₹1,500 / ₹15,000 business costs, the risk-tier cut-offs)
is defined **once** here. Notebooks and the web app import these constants,
so changing a value in one place changes it everywhere and the experiments
stay comparable.

Paths are resolved relative to *this file*, not the current working directory,
so the code works identically whether it is run from Google Drive
(``/content/drive/MyDrive/telco-churn-dashboard``), from a local clone, from
``notebooks/`` or from the project root. The root can also be overridden with
the ``CHURN_PROJECT_ROOT`` environment variable.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#: Project root = parent of the ``src/`` directory that contains this file.
PROJECT_ROOT: Path = Path(
    os.environ.get("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
ABLATION_DIR: Path = MODELS_DIR / "ablation"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
# The web app may redirect its logs (e.g. during automated tests) via CHURN_LOG_DIR.
LOGS_DIR: Path = Path(os.environ.get("CHURN_LOG_DIR", PROJECT_ROOT / "logs")).resolve()

RAW_DATA_PATH: Path = DATA_DIR / "Telco-Customer-Churn.csv"
SAMPLE_DEMO_PATH: Path = DATA_DIR / "sample_demo.csv"
DATASET_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# Serialised artefacts consumed by the inference API.
SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
ENCODER_PATH: Path = MODELS_DIR / "encoder.pkl"
FEATURE_COLUMNS_PATH: Path = MODELS_DIR / "feature_columns.pkl"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.pth"
MODEL_CONFIG_PATH: Path = MODELS_DIR / "model_config.json"
SHAP_BACKGROUND_PATH: Path = MODELS_DIR / "shap_background.npy"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Data split (fractions of the full dataset)
# ---------------------------------------------------------------------------
TRAIN_SIZE: float = 0.60
VAL_SIZE: float = 0.20
TEST_SIZE: float = 0.20

# ---------------------------------------------------------------------------
# MLP hyper-parameters (Part 4 of the brief)
# ---------------------------------------------------------------------------
HIDDEN_DIMS: tuple[int, ...] = (64, 32)
DROPOUT: float = 0.3
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4          # L2 penalty coefficient λ used by Adam
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 150
PATIENCE: int = 10                  # early-stopping patience (epochs)
OPTIMIZER_COMPARISON_EPOCHS: int = 30

# Focal-loss hyper-parameters (Part 5)
FOCAL_ALPHA: float = 0.75
FOCAL_GAMMA: float = 2.0

# ---------------------------------------------------------------------------
# Business costs (Part 6) — Indian Rupees
# ---------------------------------------------------------------------------
C_OFFER: float = 1_500.0    # cost of one retention offer / discount package
C_LOST: float = 15_000.0    # revenue lost when a churner leaves undetected
CURRENCY_SYMBOL: str = "₹"
NAIVE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Risk tiers shown in the web UI (on calibrated probabilities)
# ---------------------------------------------------------------------------
HIGH_RISK_MIN: float = 0.70     # p  > 0.70           -> High
MEDIUM_RISK_MIN: float = 0.40   # 0.40 <= p <= 0.70   -> Medium, else Low


def set_seed(seed: int = SEED) -> None:
    """
    Fix every source of randomness so experiments are exactly repeatable.

    Seeds Python's ``random`` module, NumPy's global generator and PyTorch
    (CPU and all CUDA devices), and asks cuDNN to use deterministic kernels.
    Call this **before** creating a model so that its weight initialisation is
    identical across runs, and again before each experiment in an ablation so
    that the only difference between experiments is the thing being tested.

    Parameters
    ----------
    seed : int
        The seed value (the project uses 42 everywhere).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN convolutions/reductions (slightly slower, fully repeatable).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # NumPy-only contexts (e.g. the perceptron) still work.
        pass


def get_device() -> "Any":
    """
    Return ``cuda`` when a GPU is available (e.g. a Colab T4), otherwise ``cpu``.

    Apple-silicon ``mps`` is deliberately *not* used: some BatchNorm kernels on
    MPS are non-deterministic, which would break exact reproducibility.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to (idempotent)."""
    for directory in (DATA_DIR, PROCESSED_DIR, MODELS_DIR, ABLATION_DIR,
                      PLOTS_DIR, RESULTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    """
    Read ``models/model_config.json`` (returns ``{}`` if it does not exist yet).

    The file is built up progressively by the notebooks: 04 records the
    architecture, 05 the winning imbalance strategy, 06 the calibration
    parameters and optimal threshold. The web app reads it at start-up.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def update_model_config(updates: dict[str, Any], path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    """
    Merge ``updates`` into ``model_config.json`` (read-modify-write) and return the result.

    Top-level keys in ``updates`` overwrite existing keys; nothing else is touched,
    so each notebook only owns the sections it is responsible for.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = load_model_config(path)
    config.update(updates)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=False)
    return config


def save_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as pretty-printed JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    """Read a JSON file written by :func:`save_json`."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(value: Any) -> Any:
    """Make NumPy scalars/arrays JSON-serialisable (used by :func:`save_json`)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
