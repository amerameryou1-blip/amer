from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mythos_bot.config import Settings
from mythos_bot.telemetry import MOVE_LATENCY


@dataclass(slots=True)
class ClockState:
    wtime_ms: int
    btime_ms: int
    winc_ms: int
    binc_ms: int


class AsyncUciEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self.process is not None:
            return

        self.process = await asyncio.create_subprocess_exec(
            self.settings.engine_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await self._send("uci")
        await self._wait_for("uciok", timeout=10.0)
        await self._send(f"setoption name Hash value {self.settings.engine_hash_mb}")
        await self._send(f"setoption name Threads value {self.settings.engine_threads}")
        if self.settings.weights_path:
            await self._send(f"setoption name WeightsFile value {self.settings.weights_path}")
        await self._send("isready")
        await self._wait_for("readyok", timeout=10.0)

    async def stop(self) -> None:
        if self.process is None:
            return

        with contextlib.suppress(Exception):
            await self._send("quit")
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.process.wait(), timeout=2.0)
        if self.process.returncode is None:
            self.process.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
        self.process = None

    async def new_game(self) -> None:
        await self._send("ucinewgame")
        await self._send("isready")
        await self._wait_for("readyok", timeout=10.0)

    async def get_bestmove(self, position_command: str, clocks: ClockState | None = None) -> str | None:
        async with self._lock:
            if self.process is None:
                raise RuntimeError("engine not started")

            await self._send(position_command)

            go = "go"
            if clocks is not None:
                go += (
                    f" wtime {max(0, clocks.wtime_ms)}"
                    f" btime {max(0, clocks.btime_ms)}"
                    f" winc {max(0, clocks.winc_ms)}"
                    f" binc {max(0, clocks.binc_ms)}"
                )
            else:
                go += " movetime 250"

            start = time.monotonic()
            await self._send(go)

            while True:
                line = await self._readline(timeout=self.settings.move_timeout_sec)
                if line is None:
                    return None
                if line.startswith("bestmove"):
                    MOVE_LATENCY.observe(time.monotonic() - start)
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] not in {"0000", "(none)"}:
                        return parts[1]
                    return None

    async def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("engine process unavailable")
        self.process.stdin.write((command + "\n").encode("utf-8"))
        await self.process.stdin.drain()

    async def _readline(self, timeout: float) -> str | None:
        if self.process is None or self.process.stdout is None:
            return None
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if not line:
            return None
        return line.decode("utf-8", errors="ignore").strip()

    async def _wait_for(self, token: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            line = await self._readline(timeout=remaining)
            if line is None:
                continue
            if token in line:
                return
        raise TimeoutError(f"timed out waiting for {token}")


class EnginePool:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_games)

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[AsyncUciEngine]:
        await self._semaphore.acquire()
        engine = AsyncUciEngine(self.settings)
        try:
            await engine.start()
            await engine.new_game()
            yield engine
        finally:
            await engine.stop()
            self._semaphore.release()
