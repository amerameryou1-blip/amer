#!/usr/bin/env python3
"""Upload the generated Kaggle Python files to the target GitHub repository."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


WORKDIR = Path("/kaggle/working")
REPO_OWNER = "amerameryou1-blip"
REPO_NAME = "Wjsjsjsj"
BRANCH = "main"
API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"


def get_github_token() -> str:
    """Read the GitHub token from Kaggle Secrets."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError(
            "kaggle_secrets is unavailable. This script must run inside Kaggle."
        ) from exc

    client = UserSecretsClient()
    token = client.get_secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Kaggle Secret 'GITHUB_TOKEN' was not found or is empty.")
    return token


def list_files_to_upload() -> list[Path]:
    """Upload the generated Python scripts from /kaggle/working/."""
    files = sorted(WORKDIR.glob("*.py"))
    if not files:
        raise RuntimeError("No top-level .py files were found in /kaggle/working/ to upload.")
    return files


def build_session(token: str) -> requests.Session:
    """Create a GitHub API session with the correct headers."""
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return session


def fetch_existing_sha(session: requests.Session, remote_path: str) -> str | None:
    """Return the current blob SHA if the file already exists on GitHub."""
    url = f"{API_BASE}/{quote(remote_path)}"
    response = session.get(url, params={"ref": BRANCH}, timeout=60)

    if response.status_code == 200:
        payload = response.json()
        return payload.get("sha")

    if response.status_code == 404:
        return None

    raise RuntimeError(
        f"Failed to query existing file '{remote_path}' on GitHub: "
        f"{response.status_code} {response.text}"
    )


def upload_file(session: requests.Session, file_path: Path, commit_message: str) -> None:
    """Create or update a single file via the GitHub Contents API."""
    remote_path = file_path.name
    sha = fetch_existing_sha(session, remote_path)
    url = f"{API_BASE}/{quote(remote_path)}"

    encoded_content = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    payload = {
        "message": commit_message,
        "content": encoded_content,
        "branch": BRANCH,
    }
    if sha is not None:
        payload["sha"] = sha

    response = session.put(url, json=payload, timeout=120)
    if response.status_code not in {200, 201}:
        raise RuntimeError(
            f"Failed to upload '{remote_path}': {response.status_code} {response.text}"
        )

    action = "Updated" if sha is not None else "Created"
    print(f"[SUCCESS] {action} {remote_path}")


def main() -> None:
    """Upload all generated Python files to the requested GitHub repository."""
    try:
        print("[INFO] Loading GitHub token from Kaggle Secrets...")
        token = get_github_token()
        session = build_session(token)
        files = list_files_to_upload()

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        commit_message = f"Auto-upload from Kaggle [{timestamp}]"
        print(f"[INFO] Commit message: {commit_message}")

        for file_path in files:
            print(f"[UPLOAD] {file_path.name}")
            upload_file(session, file_path, commit_message)

        print(
            f"[SUCCESS] Uploaded {len(files)} Python files to "
            f"https://github.com/{REPO_OWNER}/{REPO_NAME} on branch '{BRANCH}'."
        )
    except Exception as exc:
        print(f"[ERROR] 05_github_uploader.py failed: {exc}")
        raise


if __name__ == "__main__":
    main()
