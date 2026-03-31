from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import aiohttp

from mythos_bot.config import Settings
from mythos_bot.rate_limit import TokenBucket
from mythos_bot.telemetry import API_ERRORS, BOT_RESTARTS, TelemetryState


class LichessClient:
    def __init__(self, settings: Settings, telemetry: TelemetryState) -> None:
        self.settings = settings
        self.telemetry = telemetry
        self._session: aiohttp.ClientSession | None = None
        self._write_bucket = TokenBucket(rate=1.0, capacity=3.0)

    async def __aenter__(self) -> "LichessClient":
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self.settings.connect_timeout_sec,
            sock_read=self.settings.read_timeout_sec,
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.settings.lichess_token}",
                "User-Agent": self.settings.user_agent,
                "Accept": "application/x-ndjson",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def account(self) -> dict[str, Any]:
        return await self._get_json("/api/account")

    async def accept_challenge(self, challenge_id: str) -> None:
        await self._post(f"/api/challenge/{challenge_id}/accept", "accept_challenge")

    async def decline_challenge(self, challenge_id: str, reason: str = "generic") -> None:
        await self._post(f"/api/challenge/{challenge_id}/decline", "decline_challenge", data={"reason": reason})

    async def make_move(self, game_id: str, move: str) -> None:
        await self._post(f"/api/bot/game/{game_id}/move/{move}", "make_move")

    async def stream_incoming_events(self) -> AsyncIterator[dict[str, Any]]:
        async for event in self._stream_json("/api/stream/event", track_health=True):
            yield event

    async def stream_game(self, game_id: str) -> AsyncIterator[dict[str, Any]]:
        async for event in self._stream_json(f"/api/bot/game/stream/{game_id}", track_health=False):
            yield event

    async def _get_json(self, path: str) -> dict[str, Any]:
        session = self._require_session()
        async with session.get(self.settings.lichess_base_url + path) as response:
            response.raise_for_status()
            return await response.json()

    async def _post(self, path: str, operation: str, data: dict[str, Any] | None = None) -> None:
        session = self._require_session()
        await self._write_bucket.acquire()
        async with session.post(self.settings.lichess_base_url + path, data=data) as response:
            if response.status >= 400:
                API_ERRORS.labels(operation=operation).inc()
                response.raise_for_status()

    async def _stream_json(self, path: str, track_health: bool) -> AsyncIterator[dict[str, Any]]:
        session = self._require_session()
        backoff = self.settings.stream_backoff_min_sec

        while True:
            try:
                async with session.get(self.settings.lichess_base_url + path) as response:
                    response.raise_for_status()
                    if track_health:
                        self.telemetry.set_stream_connected(True)
                        self.telemetry.touch_stream()

                    async for raw_line in response.content:
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        if track_health:
                            self.telemetry.touch_stream()
                        yield json.loads(line)

                if track_health:
                    self.telemetry.set_stream_connected(False)
                BOT_RESTARTS.inc()
                await asyncio.sleep(backoff)
                backoff = min(self.settings.stream_backoff_max_sec, backoff * 2.0)
            except asyncio.CancelledError:
                if track_health:
                    self.telemetry.set_stream_connected(False)
                raise
            except Exception:
                if track_health:
                    self.telemetry.set_stream_connected(False)
                BOT_RESTARTS.inc()
                await asyncio.sleep(backoff)
                backoff = min(self.settings.stream_backoff_max_sec, backoff * 2.0)

    def _require_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("LichessClient must be entered before use")
        return self._session
