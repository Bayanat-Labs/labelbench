"""Default locations for downloaded and normalized source data."""

from __future__ import annotations

import os
import sys
from pathlib import Path


CACHE_ENV_VAR = "LABELBENCH_CACHE_DIR"


def user_cache_root() -> Path:
    """Return the platform's per-user cache directory."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    base = os.environ.get("XDG_CACHE_HOME")
    if base and Path(base).is_absolute():
        return Path(base)
    return Path.home() / ".cache"


def default_cache_root() -> Path:
    """Return ``$LABELBENCH_CACHE_DIR`` if set, else ``<user cache>/labelbench``."""
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return user_cache_root() / "labelbench"


def default_cache_dir() -> Path:
    """Return the default ``--cache-dir``.

    Normalized manifests are written here. Raw downloads go to the sibling
    ``raw/`` directory, so ``LABELBENCH_CACHE_DIR=data`` gives the
    ``data/normalized`` and ``data/raw`` layout used for the v0.1 runs.
    """
    return default_cache_root() / "normalized"
