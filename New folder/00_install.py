#!/usr/bin/env python3
"""Install the Kaggle runtime stack for Gemma 4 GGUF on dual T4 GPUs.

This script:
1. Ensures the required Python packages are installed.
2. Builds `llama-cpp-python` from source with CUDA enabled.
3. Builds the standalone `llama-server` binary from `ggml-org/llama.cpp`
   because the requested multi-GPU flags are exposed by the C++ server.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


WORKDIR = Path("/kaggle/working")
BIN_DIR = WORKDIR / "bin"
LLAMA_CPP_DIR = WORKDIR / "llama.cpp"
LLAMA_CPP_BUILD_DIR = LLAMA_CPP_DIR / "build"
SERVER_HINT_FILE = WORKDIR / "llama-server-path.txt"
CUDA_NVCC = Path("/usr/local/cuda/bin/nvcc")


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a subprocess with clear logging."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    pretty_command = " ".join(command)
    print(f"\n[RUN] {pretty_command}")
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None, env=merged_env)


def bootstrap_build_tools() -> None:
    """Install the basic packaging and build utilities first."""
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            "pip",
            "setuptools",
            "wheel",
            "packaging",
            "cmake",
            "ninja",
            "scikit-build-core",
        ]
    )


def get_installed_version(package_name: str) -> str | None:
    """Return an installed package version, or None if the package is missing."""
    try:
        metadata = importlib.import_module("importlib.metadata")
    except ImportError:
        metadata = importlib.import_module("importlib_metadata")

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def ensure_python_package(spec: str, package_name: str, minimum_version: str | None = None) -> None:
    """Install a package only when it is missing or below the required minimum."""
    if minimum_version:
        from packaging.version import Version

        installed = get_installed_version(package_name)
        if installed is not None and Version(installed) >= Version(minimum_version):
            print(f"[OK] {package_name} {installed} already satisfies >= {minimum_version}")
            return
    else:
        installed = get_installed_version(package_name)
        if installed is not None:
            print(f"[OK] {package_name} {installed} is already installed")
            return

    print(f"[INSTALL] {spec}")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            spec,
        ]
    )


def install_required_python_packages() -> None:
    """Install the requested Python stack in a stable order."""
    ensure_python_package("transformers>=5.5.0", "transformers", minimum_version="5.5.0")
    ensure_python_package("torch", "torch")
    ensure_python_package("accelerate", "accelerate")
    ensure_python_package("bitsandbytes", "bitsandbytes")
    ensure_python_package("pillow", "Pillow")
    ensure_python_package("timm", "timm")
    ensure_python_package("hf_transfer", "hf_transfer")
    ensure_python_package("huggingface_hub", "huggingface-hub")
    ensure_python_package("requests", "requests")
    # Gemma processors commonly rely on SentencePiece, so we install it explicitly
    # even though it was not part of the minimum list.
    ensure_python_package("sentencepiece", "sentencepiece")


def build_llama_cpp_python() -> None:
    """Build llama-cpp-python from source with CUDA support enabled."""
    env = {
        "CMAKE_ARGS": "-DGGML_CUDA=on",
        "FORCE_CMAKE": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if CUDA_NVCC.exists():
        env["CUDACXX"] = str(CUDA_NVCC)

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-binary=llama-cpp-python",
            "llama-cpp-python[server]",
        ],
        env=env,
    )


def ensure_llama_cpp_checkout() -> None:
    """Clone llama.cpp once so we can build the standalone llama-server binary."""
    if LLAMA_CPP_DIR.exists():
        print(f"[OK] Reusing existing llama.cpp checkout at {LLAMA_CPP_DIR}")
        return

    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggml-org/llama.cpp.git",
            str(LLAMA_CPP_DIR),
        ]
    )


def build_standalone_llama_server() -> Path:
    """Build the C++ llama-server binary with CUDA enabled."""
    cmake_configure = [
        "cmake",
        "-S",
        str(LLAMA_CPP_DIR),
        "-B",
        str(LLAMA_CPP_BUILD_DIR),
        "-DGGML_CUDA=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    cmake_build = [
        "cmake",
        "--build",
        str(LLAMA_CPP_BUILD_DIR),
        "--config",
        "Release",
        "-j",
    ]

    run_command(cmake_configure)
    run_command(cmake_build)

    candidates = [
        LLAMA_CPP_BUILD_DIR / "bin" / "llama-server",
        LLAMA_CPP_BUILD_DIR / "bin" / "Release" / "llama-server",
    ]
    for candidate in candidates:
        if candidate.exists():
            BIN_DIR.mkdir(parents=True, exist_ok=True)
            target = BIN_DIR / "llama-server"
            shutil.copy2(candidate, target)
            target.chmod(0o755)
            SERVER_HINT_FILE.write_text(str(target), encoding="utf-8")
            print(f"[OK] Standalone llama-server available at {target}")
            return target

    raise FileNotFoundError(
        "llama-server was not produced by the llama.cpp build. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def print_runtime_summary() -> None:
    """Print the key runtime guardrails requested for Kaggle T4 inference."""
    import torch
    import transformers

    print("\n[SUMMARY]")
    print(f"torch version: {torch.__version__}")
    print(f"transformers version: {transformers.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")

    print("Direct Transformers fallback on T4 must use:")
    print("  torch_dtype=torch.float16")
    print("  attn_implementation='sdpa'")
    print("  AutoProcessor.from_pretrained(...)")
    print("  max context 4096 for this Kaggle setup")


def main() -> None:
    """Install everything needed for the requested Kaggle workflow."""
    try:
        print("[INFO] Starting Kaggle setup install step...")
        WORKDIR.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        bootstrap_build_tools()
        install_required_python_packages()
        build_llama_cpp_python()
        ensure_llama_cpp_checkout()
        build_standalone_llama_server()
        print_runtime_summary()
        print("\n[SUCCESS] Installation and build steps completed.")
    except Exception as exc:
        print(f"\n[ERROR] 00_install.py failed: {exc}")
        raise


if __name__ == "__main__":
    main()
