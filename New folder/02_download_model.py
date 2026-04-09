#!/usr/bin/env python3
"""Download the requested Gemma 4 GGUF model from Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ID = "bartowski/google_gemma-4-26B-A4B-it-GGUF"
FILENAME = "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
MODELS_DIR = Path("/kaggle/working/models")
HF_HOME = Path("/kaggle/working/.cache/huggingface")
INPUT_ROOT = Path("/kaggle/input")


def get_huggingface_token() -> str:
    """Read the Hugging Face token from Kaggle Secrets."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError(
            "kaggle_secrets is unavailable. This script must run inside Kaggle."
        ) from exc

    client = UserSecretsClient()
    token = client.get_secret("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "Kaggle Secret 'HUGGINGFACE_TOKEN' was not found or is empty."
        )
    return token


def link_or_copy(source: Path, destination: Path) -> Path:
    """Link the model into /kaggle/working, or copy if symlinks are unavailable."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return destination

    try:
        destination.symlink_to(source)
        print(f"[OK] Symlinked cached model {source} -> {destination}")
    except OSError:
        import shutil

        shutil.copy2(source, destination)
        print(f"[OK] Copied cached model {source} -> {destination}")
    return destination


def find_cached_model() -> Path | None:
    """Search attached Kaggle inputs for a previously saved model file."""
    if not INPUT_ROOT.exists():
        return None

    exact_matches = [
        INPUT_ROOT / dataset_dir.name / "models" / FILENAME
        for dataset_dir in INPUT_ROOT.iterdir()
        if dataset_dir.is_dir()
    ]
    exact_matches.extend(
        [
            INPUT_ROOT / dataset_dir.name / FILENAME
            for dataset_dir in INPUT_ROOT.iterdir()
            if dataset_dir.is_dir()
        ]
    )

    for candidate in exact_matches:
        if candidate.exists():
            print(f"[OK] Found cached model in attached input: {candidate}")
            return candidate

    for candidate in INPUT_ROOT.rglob(FILENAME):
        if candidate.exists():
            print(f"[OK] Found cached model in attached input: {candidate}")
            return candidate

    return None


def main() -> None:
    """Download the GGUF file into /kaggle/working/models/."""
    try:
        print("[INFO] Preparing Hugging Face download...")
        HF_HOME.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(HF_HOME)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        existing_file = MODELS_DIR / FILENAME
        if existing_file.exists():
            size_gb = existing_file.stat().st_size / (1024 ** 3)
            print(f"[OK] Model already exists at {existing_file} ({size_gb:.2f} GB)")
            return

        cached_model = find_cached_model()
        if cached_model is not None:
            restored_path = link_or_copy(cached_model, existing_file)
            size_gb = restored_path.stat().st_size / (1024 ** 3)
            print(f"[SUCCESS] Reused cached model at: {restored_path}")
            print(f"[INFO] File size: {size_gb:.2f} GB")
            return

        token = get_huggingface_token()
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            token=token,
            local_dir=str(MODELS_DIR),
        )

        downloaded_path = Path(model_path)
        size_gb = downloaded_path.stat().st_size / (1024 ** 3)
        print(f"[SUCCESS] Downloaded model to: {downloaded_path}")
        print(f"[INFO] File size: {size_gb:.2f} GB")
    except Exception as exc:
        print(f"[ERROR] 02_download_model.py failed: {exc}")
        raise


if __name__ == "__main__":
    main()
