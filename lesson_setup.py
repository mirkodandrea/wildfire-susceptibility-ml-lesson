"""Runtime setup for the wildfire susceptibility lesson (Colab-specific).

Importing this module runs prepare_colab_runtime(), which installs packages
and clones the data repository when the notebook is executed in Google Colab.
Outside Colab, the import is a no-op.
"""

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
