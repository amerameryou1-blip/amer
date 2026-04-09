import base64
import importlib.util
import subprocess
import sys
from pathlib import Path

if importlib.util.find_spec("requests") is None:
    print("[SETUP] Installing requests...")
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)

import requests

WORKDIR = Path("/kaggle/working")
WORKDIR.mkdir(parents=True, exist_ok=True)

REPO_OWNER = "amerameryou1-blip"
REPO_NAME = "Wjsjsjsj"
BRANCH = "main"
BASE_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}"
FILE_MAP = {
    "00_install.py": "00_install_v2.py",
    "01_auth.py": "01_auth.py",
    "02_download_model.py": "02_download_model.py",
    "03_run_server.py": "03_run_server_v2.py",
    "04_inference_test.py": "04_inference_test.py",
    "05_github_uploader.py": "05_github_uploader.py",
}

print("[INFO] Cache behavior:")
print("[INFO] 1) Reuse current-session files already in /kaggle/working")
print("[INFO] 2) Reuse any attached Kaggle input dataset that already contains the model/server")
print("[INFO] 3) Only download/build missing pieces")

headers = {}
try:
    from kaggle_secrets import UserSecretsClient

    github_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
except Exception as secret_exc:
    print(f"[WARN] Could not load GITHUB_TOKEN for raw file fetches: {secret_exc}")

for destination_name, source_name in FILE_MAP.items():
    url = f"{BASE_URL}/{source_name}"
    destination = WORKDIR / destination_name
    print(f"[FETCH-START] {destination_name} <- {url}")
    response = requests.get(url, headers=headers, timeout=120)
    if response.status_code == 200:
        destination.write_text(response.text, encoding="utf-8")
    else:
        print(
            f"[WARN] Raw fetch failed for {source_name} with status {response.status_code}; "
            "trying GitHub Contents API fallback."
        )
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{source_name}"
        api_headers = {"Accept": "application/vnd.github+json"}
        if "Authorization" in headers:
            api_headers["Authorization"] = headers["Authorization"]
        api_response = requests.get(api_url, headers=api_headers, params={"ref": BRANCH}, timeout=120)
        api_response.raise_for_status()
        payload = api_response.json()
        content = base64.b64decode(payload["content"]).decode("utf-8")
        destination.write_text(content, encoding="utf-8")
    print(f"[FETCH-DONE] Saved {destination_name} to {destination}")


def print_log_tail(log_path: Path, max_lines: int = 120) -> None:
    if not log_path.exists():
        print(f"[WARN] Log file not found: {log_path}")
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"[LOG-TAIL] Showing last {min(len(lines), max_lines)} lines from {log_path}")
    for line in lines[-max_lines:]:
        print(line)


for filename in FILE_MAP:
    script_path = WORKDIR / filename
    log_path = WORKDIR / f"{filename}.log"
    print(f"[RUN-START] {filename}")
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        print(f"[ERROR] {filename} failed with exit code {result.returncode}")
        print_log_tail(log_path)
        raise RuntimeError(f"{filename} failed. Full log: {log_path}")
    print_log_tail(log_path, max_lines=40)
    print(f"[RUN-DONE] {filename}")

print("[SUCCESS] Kaggle Gemma 4 setup finished.")
