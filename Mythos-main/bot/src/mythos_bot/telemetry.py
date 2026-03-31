from __future__ import annotations

import time
from dataclasses import dataclass, field

from prometheus_client import Counter, Gauge, Histogram


GAMES_ACTIVE = Gauge("mythos_active_games", "Number of active Lichess games")
STREAM_CONNECTED = Gauge("mythos_stream_connected", "Incoming event stream connection status")
STREAM_AGE = Gauge("mythos_stream_last_event_age_seconds", "Seconds since the last stream event")
READY = Gauge("mythos_ready", "Readiness status")
BOT_RESTARTS = Counter("mythos_stream_restarts_total", "Number of stream reconnects")
API_ERRORS = Counter("mythos_api_errors_total", "Lichess API errors", ["operation"])
MOVE_LATENCY = Histogram(
    "mythos_move_latency_seconds",
    "Time spent waiting for engine bestmove",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30),
)
GAME_RESULTS = Counter("mythos_game_results_total", "Completed games by result", ["result"])


@dataclass(slots=True)
class TelemetryState:
    ready: bool = False
    stream_connected: bool = False
    last_event_monotonic: float = field(default_factory=time.monotonic)
    active_games: int = 0

    def set_ready(self, ready: bool) -> None:
        self.ready = ready
        READY.set(1 if ready else 0)

    def set_stream_connected(self, connected: bool) -> None:
        self.stream_connected = connected
        STREAM_CONNECTED.set(1 if connected else 0)

    def touch_stream(self) -> None:
        self.last_event_monotonic = time.monotonic()
        STREAM_AGE.set(0.0)

    def update_stream_age(self) -> float:
        age = max(0.0, time.monotonic() - self.last_event_monotonic)
        STREAM_AGE.set(age)
        return age

    def game_started(self) -> None:
        self.active_games += 1
        GAMES_ACTIVE.set(self.active_games)

    def game_finished(self, result: str) -> None:
        self.active_games = max(0, self.active_games - 1)
        GAMES_ACTIVE.set(self.active_games)
        GAME_RESULTS.labels(result=result).inc()
