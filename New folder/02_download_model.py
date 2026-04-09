#!/usr/bin/env python3
"""Download the requested Gemma 4 GGUF model from Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ID = "bartowski/google_gemma-4-26B-A4B-it-GGUF"
FILENAME = "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
MODELS_DIR = Path("/kaggle/working/models")
HF_HOME = Path("/kaggle/working/.cache/huggingface")


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


def main() -> None:
    """Download the GGUF file into /kaggle/working/models/."""
    try:
        print("[INFO] Preparing Hugging Face download...")
        token = get_huggingface_token()

        HF_HOME.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(HF_HOME)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        from huggingface_hub import hf_hub_download

        existing_file = MODELS_DIR / FILENAME
        if existing_file.exists():
            size_gb = existing_file.stat().st_size / (1024 ** 3)
            print(f"[OK] Model already exists at {existing_file} ({size_gb:.2f} GB)")
            return

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
