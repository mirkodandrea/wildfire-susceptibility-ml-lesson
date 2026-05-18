"""Runtime setup and shared imports for the wildfire susceptibility lesson."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/mirkodandrea/wildfire-susceptibility-ml-lesson.git"
REPO_DIR = Path("/content/wildfire-susceptibility-ml-lesson")


def running_in_colab() -> bool:
    """Return True when the lesson is running inside Google Colab."""
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def prepare_colab_runtime() -> None:
    """Install dependencies and clone the repository when running in Colab."""
    if not running_in_colab():
        return

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "rasterio",
            "scikit-learn",
            "matplotlib",
            "pandas",
            "numpy",
            "ipython",
        ]
    )

    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    os.chdir(REPO_DIR)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    print(f"Working directory: {Path.cwd()}")


prepare_colab_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from IPython.display import Markdown, display
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split

from utils import (
    combined_fire_mask,
    full_period_table,
    pixel_frame,
    sampled_period_table,
)

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)

__all__ = [
    "Path",
    "plt",
    "np",
    "pd",
    "rasterio",
    "Markdown",
    "display",
    "BoundaryNorm",
    "ListedColormap",
    "Patch",
    "RandomForestClassifier",
    "permutation_importance",
    "ConfusionMatrixDisplay",
    "PrecisionRecallDisplay",
    "RocCurveDisplay",
    "average_precision_score",
    "precision_score",
    "recall_score",
    "roc_auc_score",
    "GridSearchCV",
    "PredefinedSplit",
    "train_test_split",
    "combined_fire_mask",
    "full_period_table",
    "pixel_frame",
    "sampled_period_table",
]
