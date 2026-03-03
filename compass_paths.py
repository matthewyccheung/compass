from __future__ import annotations

import os
from pathlib import Path


DEFAULT_RESULTS_DIR = "/scratch/yc130/compass_results"


def get_results_base_dir() -> Path:
    """
    Base directory for all outputs that should not be committed to the repo.

    Override by setting `COMPASS_RESULTS_DIR`.
    """
    return Path(os.environ.get("COMPASS_RESULTS_DIR", DEFAULT_RESULTS_DIR)).expanduser()


def ensure_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot create results directory at '{path}'. "
            f"Set COMPASS_RESULTS_DIR to a writable location."
        ) from e
    return path


def get_results_dir(*parts: str) -> Path:
    return ensure_dir(get_results_base_dir().joinpath(*parts))

