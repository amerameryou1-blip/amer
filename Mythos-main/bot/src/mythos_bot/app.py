from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from typing import Any

from mythos_bot.config import Settings
from mythos_bot.engine_pool import EnginePool
from mythos_bot.game_session import GameSession
from mythos_bot.health import HealthServer
from mythos_bot.lichess_client import LichessClient
from mythos_bot.telemetry import API_ERRORS, TelemetryState


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class MythosBotApp:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.telemetry = TelemetryState()
        self.engine_pool = EnginePool(settings)
        self.health_server = HealthServer(self.telemetry)
        self.bot_id = ""
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.stopping = asyncio.Event()
        self.logger = logging.getLogger("mythos_bot")

    async def run(self) -> None:
        await self.health_server.start(self.settings.metrics_host, self.settings.metrics_port)
        try:
            async with LichessClient(self.settings, self.telemetry) as client:
                account = await client.account()
                self.bot_id = account.get("id", "")
                if account.get("title") != "BOT":
                    raise RuntimeError(f"account {self.bot_id} is not a BOT account")

                self.telemetry.set_ready(True)
                await self._run_event_loop(client)
        finally:
            self.telemetry.set_ready(False)
            await self.health_server.stop()

    async def _run_event_loop(self, client: LichessClient) -> None:
        while not self.stopping.is_set():
            try:
                async for event in client.stream_incoming_events():
                    if self.stopping.is_set():
                        break
                    await self._dispatch_event(client, event)
            except asyncio.CancelledError:
                raise
            except Exception:
                API_ERRORS.labels(operation="event_stream").inc()
                self.logger.exception("incoming event stream failed")
                await asyncio.sleep(self.settings.stream_backoff_min_sec)

    async def _dispatch_event(self, client: LichessClient, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type == "challenge":
            await self._handle_challenge(client, event.get("challenge", {}))
        elif event_type == "gameStart":
            await self._handle_game_start(client, event.get("game", {}))
        elif event_type == "gameFinish":
            game_id = event.get("game", {}).get("id", "")
            task = self.tasks.pop(game_id, None)
            if task is not None:
                task.cancel()

        self._cleanup_finished_tasks()

    async def _handle_challenge(self, client: LichessClient, challenge: dict[str, Any]) -> None:
        challenge_id = challenge.get("id", "")
        if not challenge_id:
            return

        if len(self.tasks) >= self.settings.max_concurrent_games:
            await client.decline_challenge(challenge_id, reason="later")
            return

        variant = challenge.get("variant", {}).get("key", "standard")
        speed = challenge.get("speed", "blitz")
        time_control = challenge.get("timeControl", {})
        initial = int(time_control.get("limit", 0))
        increment = int(time_control.get("increment", 0))

        if self.settings.accepted_variants and variant not in self.settings.accepted_variants:
            await client.decline_challenge(challenge_id, reason="variant")
            return
        if self.settings.accepted_speeds and speed not in self.settings.accepted_speeds:
            await client.decline_challenge(challenge_id, reason="timeControl")
            return
        if initial < self.settings.min_initial_seconds or initial > self.settings.max_initial_seconds:
            await client.decline_challenge(challenge_id, reason="timeControl")
            return
        if increment > self.settings.max_increment_seconds:
            await client.decline_challenge(challenge_id, reason="timeControl")
            return

        await client.accept_challenge(challenge_id)

    async def _handle_game_start(self, client: LichessClient, game: dict[str, Any]) -> None:
        game_id = game.get("gameId") or game.get("id")
        if not game_id or game_id in self.tasks:
            return

        session = GameSession(client, self.settings, self.telemetry, self.engine_pool, game_id, self.bot_id)
        self.tasks[game_id] = asyncio.create_task(session.run(), name=f"game:{game_id}")

    def _cleanup_finished_tasks(self) -> None:
        finished = [game_id for game_id, task in self.tasks.items() if task.done()]
        for game_id in finished:
            task = self.tasks.pop(game_id)
            if task.cancelled():
                continue
            exception = task.exception()
            if exception is not None:
                self.logger.error("game task %s failed: %s", game_id, exception)

    async def shutdown(self) -> None:
        self.stopping.set()
        self.telemetry.set_ready(False)
        for task in self.tasks.values():
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()


async def _main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    app = MythosBotApp(settings)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    runner = asyncio.create_task(app.run())
    stopper = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait({runner, stopper}, return_when=asyncio.FIRST_COMPLETED)

    if stopper in done:
        await app.shutdown()
    if runner in done:
        await runner
    else:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
    for task in pending:
        task.cancel()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
