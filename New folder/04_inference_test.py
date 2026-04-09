#!/usr/bin/env python3
"""Run a small OpenAI-compatible chat test against the local llama-server."""

from __future__ import annotations

import time
from pathlib import Path

import requests


HOST = "127.0.0.1"
PORT = 8080
CHAT_URL = f"http://{HOST}:{PORT}/v1/chat/completions"
HEALTH_URLS = [
    f"http://{HOST}:{PORT}/v1/health",
    f"http://{HOST}:{PORT}/health",
    f"http://{HOST}:{PORT}/v1/models",
]
LOG_FILE = Path("/kaggle/working/llama-server.log")


def read_log_tail(max_lines: int = 120) -> str:
    """Return the tail of the server log if the request fails."""
    if not LOG_FILE.exists():
        return "(llama-server log does not exist yet)"
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def wait_for_server(timeout_seconds: int = 300) -> None:
    """Wait until one of the health endpoints is reachable."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        for url in HEALTH_URLS:
            try:
                response = requests.get(url, timeout=5)
                if response.ok:
                    print(f"[OK] Server health check passed via {url}")
                    return
            except requests.RequestException:
                continue

        elapsed = int(time.time() - start_time)
        print(f"[WAIT] Server not ready yet... ({elapsed}s elapsed)")
        time.sleep(5)

    raise TimeoutError(
        "llama-server never became reachable for inference.\n\n"
        f"Recent log tail:\n{read_log_tail()}"
    )


def main() -> None:
    """Send a short chat request and print the model response."""
    try:
        wait_for_server()

        payload = {
            "model": "google_gemma-4-26B-A4B-it-Q4_K_M",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise assistant running in a Kaggle validation test.",
                },
                {
                    "role": "user",
                    "content": "Reply with one short sentence confirming the llama.cpp server works.",
                },
            ],
            "temperature": 0.2,
            "max_tokens": 64,
            "stream": False,
        }

        print("[INFO] Sending test request to llama-server...")
        response = requests.post(CHAT_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned by server: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("text")
        if not content:
            raise RuntimeError(f"No text content returned by server: {data}")

        print("[SUCCESS] Inference response:")
        print(content.strip())

        usage = data.get("usage")
        if usage:
            print("[INFO] Token usage:")
            print(usage)
    except Exception as exc:
        print(f"[ERROR] 04_inference_test.py failed: {exc}")
        print("\n[DEBUG] Recent llama-server log tail:")
        print(read_log_tail())
        raise


if __name__ == "__main__":
    main()
