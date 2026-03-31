from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from mythos_bot.config import Settings
from mythos_bot.engine_pool import ClockState, EnginePool
from mythos_bot.lichess_client import LichessClient
from mythos_bot.telemetry import TelemetryState


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GameContext:
    game_id: str
    bot_id: str
    initial_fen: str = "startpos"
    is_white: bool = True
    opponent_name: str = "unknown"
    last_position_command: str = ""


class GameSession:
    def __init__(
        self,
        client: LichessClient,
        settings: Settings,
        telemetry: TelemetryState,
        engine_pool: EnginePool,
        game_id: str,
        bot_id: str,
    ) -> None:
        self.client = client
        self.settings = settings
        self.telemetry = telemetry
        self.engine_pool = engine_pool
        self.context = GameContext(game_id=game_id, bot_id=bot_id)
        self._stop = asyncio.Event()

    async def run(self) -> None:
        self.telemetry.game_started()
        result = "unknown"

        try:
            async with self.engine_pool.acquire() as engine:
                async for event in self.client.stream_game(self.context.game_id):
                    if self._stop.is_set():
                        break

                    event_type = event.get("type")
                    if event_type == "gameFull":
                        await self._handle_game_full(engine, event)
                    elif event_type == "gameState":
                        state_result = await self._handle_game_state(engine, event)
                        if state_result is not None:
                            result = state_result
                            break
        except asyncio.CancelledError:
            result = "cancelled"
            raise
        except Exception:
            LOGGER.exception("game %s failed", self.context.game_id)
            result = "error"
        finally:
            self.telemetry.game_finished(result)

    def stop(self) -> None:
        self._stop.set()

    async def _handle_game_full(self, engine, event: dict[str, Any]) -> None:
        white = event.get("white", {})
        black = event.get("black", {})
        self.context.is_white = white.get("id", "").lower() == self.context.bot_id.lower()
        self.context.opponent_name = (
            black.get("name", black.get("id", "unknown"))
            if self.context.is_white
            else white.get("name", white.get("id", "unknown"))
        )
        self.context.initial_fen = event.get("initialFen", "startpos") or "startpos"
        state = event.get("state", {})
        await self._handle_play(engine, state)

    async def _handle_game_state(self, engine, event: dict[str, Any]) -> str | None:
        status = event.get("status", "started")
        if status != "started":
            winner = event.get("winner")
            if winner is None:
                return "draw"
            won = (winner == "white" and self.context.is_white) or (winner == "black" and not self.context.is_white)
            return "win" if won else "loss"

        await self._handle_play(engine, event)
        return None

    async def _handle_play(self, engine, state: dict[str, Any]) -> None:
        moves = state.get("moves", "").strip()
        move_list = moves.split() if moves else []
        is_white_turn = len(move_list) % 2 == 0
        if is_white_turn != self.context.is_white:
            return

        if self.context.initial_fen == "startpos":
            position = "position startpos"
        else:
            position = f"position fen {self.context.initial_fen}"
        if moves:
            position += f" moves {moves}"

        if position == self.context.last_position_command:
            return

        clocks = ClockState(
            wtime_ms=int(state.get("wtime", 60000)),
            btime_ms=int(state.get("btime", 60000)),
            winc_ms=int(state.get("winc", 0)),
            binc_ms=int(state.get("binc", 0)),
        )
        bestmove = await engine.get_bestmove(position, clocks=clocks)
        if bestmove:
            await self.client.make_move(self.context.game_id, bestmove)
            self.context.last_position_command = position
