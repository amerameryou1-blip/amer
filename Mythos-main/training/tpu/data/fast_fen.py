from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


CPP_EXT_DIR = Path(__file__).resolve().parent / "cpp_ext"
PROJECT_ROOT = CPP_EXT_DIR.parents[3]


def _decode_single_fen_py(fen: str) -> tuple[np.ndarray, bool] | None:
    features = np.zeros(768, dtype=np.float32)
    piece_to_idx = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    parts = fen.split()
    if not parts:
        return None

    placement = parts[0]
    square = 56
    saw_white_king = False
    saw_black_king = False

    for char in placement:
        if char == "/":
            square -= 16
        elif char.isdigit():
            square += int(char)
        else:
            plane = piece_to_idx.get(char)
            if plane is None or not 0 <= square < 64:
                return None
            features[plane * 64 + square] = 1.0
            square += 1
            saw_white_king = saw_white_king or char == "K"
            saw_black_king = saw_black_king or char == "k"

    if not saw_white_king or not saw_black_king:
        return None

    black_to_move = len(parts) > 1 and parts[1] == "b"
    return features, black_to_move


def _decode_binary_records_py(binary_records: bytes, flip_to_stm: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
    offset = 0
    features: list[np.ndarray] = []
    targets: list[list[float]] = []

    while offset + 3 <= len(binary_records):
        fen_length = int.from_bytes(binary_records[offset : offset + 2], byteorder="little", signed=False)
        offset += 2
        if offset + fen_length + 1 > len(binary_records):
            break

        fen = binary_records[offset : offset + fen_length].decode("utf-8", errors="ignore")
        offset += fen_length
        target = int.from_bytes(binary_records[offset : offset + 1], byteorder="little", signed=True)
        offset += 1

        decoded = _decode_single_fen_py(fen)
        if decoded is None:
            continue

        feature_vec, black_to_move = decoded
        if flip_to_stm and black_to_move:
            target = -target

        features.append(feature_vec)
        targets.append([float(target)])

    if not features:
        return (
            torch.empty((0, 768), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
            0,
        )

    feature_tensor = torch.from_numpy(np.stack(features, axis=0))
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return feature_tensor, target_tensor, feature_tensor.shape[0]


def _load_cpp_module():
    from training.tpu.data.cpp_ext import _fast_fen  # type: ignore

    return _fast_fen


def ensure_fast_fen_built(verbose: bool = False) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing else f"{PROJECT_ROOT}{os.pathsep}{existing}"
    command = [sys.executable, "setup.py", "build_ext", "--inplace"]
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    subprocess.run(command, cwd=str(CPP_EXT_DIR), env=env, check=True, stdout=stdout, stderr=stderr)


def decode_binary_records(
    binary_records: bytes | memoryview,
    flip_to_stm: bool = True,
    prefer_cpp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    payload = binary_records.tobytes() if isinstance(binary_records, memoryview) else binary_records

    if prefer_cpp:
        try:
            module = _load_cpp_module()
            features, targets, count = module.decode_binary_records(np.frombuffer(payload, dtype=np.uint8), flip_to_stm)
            return torch.from_numpy(features), torch.from_numpy(targets), int(count)
        except Exception:
            pass

    return _decode_binary_records_py(payload, flip_to_stm=flip_to_stm)
