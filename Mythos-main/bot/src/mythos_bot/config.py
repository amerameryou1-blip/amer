from __future__ import annotations

import os
from dataclasses import dataclass


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(slots=True)
class Settings:
    lichess_token: str
    lichess_base_url: str
    engine_path: str
    weights_path: str
    engine_hash_mb: int
    engine_threads: int
    max_concurrent_games: int
    move_timeout_sec: float
    stream_backoff_min_sec: float
    stream_backoff_max_sec: float
    connect_timeout_sec: float
    read_timeout_sec: float
    user_agent: str
    metrics_host: str
    metrics_port: int
    accepted_variants: tuple[str, ...]
    accepted_speeds: tuple[str, ...]
    min_initial_seconds: int
    max_initial_seconds: int
    max_increment_seconds: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("LICHESS_TOKEN", "").strip()
        if not token:
            raise ValueError("LICHESS_TOKEN is required")

        return cls(
            lichess_token=token,
            lichess_base_url=os.getenv("LICHESS_BASE_URL", "https://lichess.org").rstrip("/"),
            engine_path=os.getenv("ENGINE_PATH", "/app/engine/mythos"),
            weights_path=os.getenv("WEIGHTS_PATH", "/app/artifacts/weights.bin"),
            engine_hash_mb=int(os.getenv("ENGINE_HASH_MB", "64")),
            engine_threads=int(os.getenv("ENGINE_THREADS", "1")),
            max_concurrent_games=int(os.getenv("MAX_CONCURRENT_GAMES", "4")),
            move_timeout_sec=float(os.getenv("MOVE_TIMEOUT_SEC", "20")),
            stream_backoff_min_sec=float(os.getenv("STREAM_BACKOFF_MIN_SEC", "1")),
            stream_backoff_max_sec=float(os.getenv("STREAM_BACKOFF_MAX_SEC", "30")),
            connect_timeout_sec=float(os.getenv("CONNECT_TIMEOUT_SEC", "15")),
            read_timeout_sec=float(os.getenv("READ_TIMEOUT_SEC", "90")),
            user_agent=os.getenv("USER_AGENT", "MythosBot/10"),
            metrics_host=os.getenv("METRICS_HOST", "0.0.0.0"),
            metrics_port=int(os.getenv("METRICS_PORT", "8080")),
            accepted_variants=_split_csv(os.getenv("ACCEPTED_VARIANTS", "standard")),
            accepted_speeds=_split_csv(os.getenv("ACCEPTED_SPEEDS", "bullet,blitz,rapid")),
            min_initial_seconds=int(os.getenv("MIN_INITIAL_SECONDS", "0")),
            max_initial_seconds=int(os.getenv("MAX_INITIAL_SECONDS", "1800")),
            max_increment_seconds=int(os.getenv("MAX_INCREMENT_SECONDS", "30")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
