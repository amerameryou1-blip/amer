"""
Filesystem helpers for Territorial.io bot.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Args:
        path (str | Path): Directory path.

    Returns:
        Path: Resolved directory path.
    """

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_pickle_save(obj: Any, path: str | Path) -> None:
    """
    Atomically write a pickle file by saving to a temporary path first.

    Args:
        obj (Any): Python object to serialize.
        path (str | Path): Destination file path.
    """

    destination = Path(path)
    ensure_dir(destination.parent)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(destination)


def safe_pickle_load(path: str | Path) -> Any | None:
    """
    Load a pickle file, returning None on any failure.

    Args:
        path (str | Path): Source file path.

    Returns:
        Any | None: Loaded object or None if loading failed.
    """

    source = Path(path)
    if not source.exists():
        return None

    try:
        with source.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def get_latest_checkpoint(checkpoint_dir: str | Path) -> str | None:
    """
    Return the most recently modified pickle checkpoint in a directory.

    Args:
        checkpoint_dir (str | Path): Checkpoint directory path.

    Returns:
        str | None: Latest checkpoint path string, or None if absent.
    """

    directory = Path(checkpoint_dir)
    if not directory.exists():
        return None

    candidates = sorted(directory.glob("*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return str(candidates[0])


def list_worker_checkpoints(checkpoint_dir: str | Path, num_workers: int) -> list[str]:
    """
    Return a list of expected worker checkpoint paths that currently exist.

    Args:
        checkpoint_dir (str | Path): Checkpoint directory path.
        num_workers (int): Number of worker IDs to scan.

    Returns:
        list[str]: Existing worker checkpoint paths.
    """

    directory = Path(checkpoint_dir)
    checkpoints: list[str] = []
    for worker_id in range(num_workers):
        worker_path = directory / f"worker_{worker_id}_q_table.pkl"
        if worker_path.exists():
            checkpoints.append(str(worker_path))
    return checkpoints
