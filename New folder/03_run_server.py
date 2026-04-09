#!/usr/bin/env python3
"""Launch llama-server for Gemma 4 GGUF on Kaggle dual T4 GPUs."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


WORKDIR = Path("/kaggle/working")
MODELS_DIR = WORKDIR / "models"
MODEL_PATH = MODELS_DIR / "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
SERVER_HINT_FILE = WORKDIR / "llama-server-path.txt"
LOG_FILE = WORKDIR / "llama-server.log"
PID_FILE = WORKDIR / "llama-server.pid"
SERVER_CONFIG_FILE = WORKDIR / "llama_cpp_server_config.json"
INPUT_ROOT = Path("/kaggle/input")
HOST = "127.0.0.1"
PORT = 8080
HEALTH_URLS = [
    f"http://{HOST}:{PORT}/v1/health",
    f"http://{HOST}:{PORT}/health",
    f"http://{HOST}:{PORT}/v1/models",
]


def read_log_tail(max_lines: int = 120) -> str:
    """Return the tail of the server log for easier debugging."""
    if not LOG_FILE.exists():
        return "(log file does not exist yet)"

    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def find_llama_server_binary() -> Path:
    """Find the standalone llama-server binary built during installation."""
    candidates = []

    if SERVER_HINT_FILE.exists():
        hinted = Path(SERVER_HINT_FILE.read_text(encoding="utf-8").strip())
        candidates.append(hinted)

    which_path = shutil.which("llama-server")
    if which_path:
        candidates.append(Path(which_path))

    candidates.extend(
        [
            WORKDIR / "bin" / "llama-server",
            WORKDIR / "llama.cpp" / "build" / "bin" / "llama-server",
            WORKDIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-server",
        ]
    )

    if INPUT_ROOT.exists():
        patterns = [
            "*/bin/llama-server",
            "*/llama-server",
            "*/llama.cpp/build/bin/llama-server",
            "*/llama.cpp/build/bin/Release/llama-server",
        ]
        for pattern in patterns:
            for candidate in INPUT_ROOT.glob(pattern):
                candidates.append(candidate)

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find llama-server. Run 00_install.py first and confirm the build succeeded."
    )


def get_optional_mmproj_path() -> Path | None:
    """Use mmproj automatically if the user later adds it to the models directory."""
    candidates = [
        MODELS_DIR / "mmproj-google_gemma-4-26B-A4B-it-f16.gguf",
        MODELS_DIR / "mmproj-google_gemma-4-26B-A4B-it-bf16.gguf",
    ]
    if INPUT_ROOT.exists():
        for pattern in [
            "*/models/mmproj-google_gemma-4-26B-A4B-it-f16.gguf",
            "*/models/mmproj-google_gemma-4-26B-A4B-it-bf16.gguf",
            "*/mmproj-google_gemma-4-26B-A4B-it-f16.gguf",
            "*/mmproj-google_gemma-4-26B-A4B-it-bf16.gguf",
        ]:
            for candidate in INPUT_ROOT.glob(pattern):
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_model_path() -> Path:
    """Find the GGUF in working storage first, then attached Kaggle inputs."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    if INPUT_ROOT.exists():
        patterns = [
            "*/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
            "*/google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
        ]
        for pattern in patterns:
            for candidate in INPUT_ROOT.glob(pattern):
                if candidate.exists():
                    print(f"[OK] Using cached model from attached input: {candidate}")
                    return candidate

    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH} and no cached input copy was found. "
        "Run 02_download_model.py first."
    )


def is_server_ready() -> bool:
    """Check whether the server is answering on any common health endpoint."""
    for url in HEALTH_URLS:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return True
        except requests.RequestException:
            continue
    return False


def process_is_alive(pid: int) -> bool:
    """Check whether a detached process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_stale_process_if_needed() -> None:
    """Clean up stale pid files left over from earlier Kaggle runs."""
    if not PID_FILE.exists():
        return

    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        return

    if process_is_alive(pid):
        if is_server_ready():
            print(f"[OK] llama-server is already running with PID {pid}")
            return

        print(f"[INFO] Found stale/unready llama-server process {pid}; stopping it...")
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except OSError:
            pass

    PID_FILE.unlink(missing_ok=True)


def build_command(server_binary: Path) -> list[str]:
    """Construct the exact llama-server command requested for Kaggle dual T4s."""
    resolved_model_path = find_model_path()
    command = [
        str(server_binary),
        "--model",
        str(resolved_model_path),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--n-gpu-layers",
        "999",
        "--tensor-split",
        "50,50",
        "--split-mode",
        "layer",
        "--ctx-size",
        "4096",
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "--parallel",
        "1",
        "--ubatch",
        "512",
        "--jinja",
    ]

    # Gemma 4 can operate in text-only mode without mmproj, but if the user later
    # places the projector file in /kaggle/working/models/ we wire it in automatically.
    mmproj_path = get_optional_mmproj_path()
    if mmproj_path is not None:
        command.extend(["--mmproj", str(mmproj_path)])
        print(f"[INFO] Using mmproj file: {mmproj_path}")
    else:
        print("[INFO] No mmproj file found; starting in text-only mode.")

    return command


def resolve_q4_0_cache_type() -> int:
    """Resolve the runtime enum value for GGML q4_0 cache quantization."""
    import llama_cpp

    for attr_name in ("GGML_TYPE_Q4_0", "ggml_type_q4_0"):
        if hasattr(llama_cpp, attr_name):
            return int(getattr(llama_cpp, attr_name))

    raise RuntimeError(
        "Could not resolve GGML_TYPE_Q4_0 from llama_cpp. "
        "The installed llama-cpp-python build may be incompatible."
    )


def build_python_server_command() -> list[str]:
    """Build a llama_cpp.server command with a JSON config file."""
    import llama_cpp

    resolved_model_path = find_model_path()
    q4_0_type = resolve_q4_0_cache_type()

    config = {
        "host": HOST,
        "port": PORT,
        "interrupt_requests": True,
        "models": [
            {
                "model": str(resolved_model_path),
                "model_alias": "google_gemma-4-26B-A4B-it-Q4_K_M",
                "n_gpu_layers": -1,
                "split_mode": int(llama_cpp.LLAMA_SPLIT_MODE_LAYER),
                "tensor_split": [0.5, 0.5],
                "n_ctx": 4096,
                "n_batch": 512,
                "n_ubatch": 512,
                "offload_kqv": True,
                "flash_attn": False,
                "type_k": q4_0_type,
                "type_v": q4_0_type,
                "verbose": True,
            }
        ],
    }

    mmproj_path = get_optional_mmproj_path()
    if mmproj_path is not None:
        # llama-cpp-python's server uses clip_model_path for multimodal helpers.
        config["models"][0]["clip_model_path"] = str(mmproj_path)
        print(f"[INFO] Using mmproj/clip model file: {mmproj_path}")
    else:
        print("[INFO] No mmproj file found; starting in text-only mode.")

    SERVER_CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote server config to {SERVER_CONFIG_FILE}")

    return [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--config_file",
        str(SERVER_CONFIG_FILE),
    ]


def launch_server() -> int:
    """Start llama-server as a detached subprocess and return its PID."""
    try:
        server_binary = find_llama_server_binary()
        command = build_command(server_binary)
        print("[INFO] Using standalone llama-server binary.")
    except Exception as binary_exc:
        print(f"[WARN] Standalone llama-server unavailable: {binary_exc}")
        print("[INFO] Falling back to python -m llama_cpp.server.")
        command = build_python_server_command()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(LOG_FILE, "ab")

    print("[INFO] Launching llama-server...")
    print("[INFO] Command:")
    print(" ".join(command))

    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(WORKDIR),
        env=env,
        start_new_session=True,
    )
    log_handle.close()
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    return process.pid


def wait_until_ready(pid: int, timeout_seconds: int = 900) -> None:
    """Wait for the server to finish loading the model."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if is_server_ready():
            print(f"[SUCCESS] llama-server is ready on http://{HOST}:{PORT}")
            print(f"[INFO] PID: {pid}")
            print(f"[INFO] Log file: {LOG_FILE}")
            return

        if not process_is_alive(pid):
            raise RuntimeError(
                "llama-server exited before becoming ready.\n\n"
                f"Recent log tail:\n{read_log_tail()}"
            )

        elapsed = int(time.time() - start_time)
        print(f"[WAIT] llama-server still loading... ({elapsed}s elapsed)")
        time.sleep(10)

    raise TimeoutError(
        "Timed out waiting for llama-server to become ready.\n\n"
        f"Recent log tail:\n{read_log_tail()}"
    )


def main() -> None:
    """Launch the server unless it is already healthy."""
    try:
        stop_stale_process_if_needed()
        if PID_FILE.exists() and is_server_ready():
            print(f"[OK] llama-server is already healthy at http://{HOST}:{PORT}")
            return

        pid = launch_server()
        wait_until_ready(pid)
    except Exception as exc:
        print(f"[ERROR] 03_run_server.py failed: {exc}")
        raise


if __name__ == "__main__":
    main()
