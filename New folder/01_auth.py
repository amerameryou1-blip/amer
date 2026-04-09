#!/usr/bin/env python3
"""Authenticate to Hugging Face from Kaggle Secrets."""

from __future__ import annotations

import os
from pathlib import Path


HF_HOME = Path("/kaggle/working/.cache/huggingface")
MODEL_ID = "google/gemma-4-26B-A4B-it"


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
    """Authenticate and sanity-check the Gemma 4 processor path."""
    try:
        print("[INFO] Loading Hugging Face token from Kaggle Secrets...")
        token = get_huggingface_token()

        HF_HOME.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(HF_HOME)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        from huggingface_hub import login
        from packaging.version import Version
        import torch
        import transformers
        from transformers import AutoProcessor

        if Version(transformers.__version__) < Version("5.5.0"):
            raise RuntimeError(
                f"transformers {transformers.__version__} is too old; need >= 5.5.0 for Gemma 4."
            )

        login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
        print("[OK] Hugging Face login succeeded.")

        # We intentionally use AutoProcessor instead of AutoTokenizer because Gemma 4
        # is a multimodal model family and the user explicitly requested this path.
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
        print(f"[OK] AutoProcessor loaded successfully: {processor.__class__.__name__}")

        # These are the safe direct-Transformers settings for Kaggle T4 GPUs.
        print("[INFO] T4-safe direct Transformers settings:")
        print(f"  torch dtype: {torch.float16}")
        print("  attn_implementation: sdpa")
        print("  tokenizer path: AutoProcessor (not AutoTokenizer)")
    except Exception as exc:
        print(f"[ERROR] 01_auth.py failed: {exc}")
        raise


if __name__ == "__main__":
    main()
