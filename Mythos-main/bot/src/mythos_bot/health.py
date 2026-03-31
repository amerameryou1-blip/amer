from __future__ import annotations

from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from mythos_bot.telemetry import TelemetryState


class HealthServer:
    def __init__(self, telemetry: TelemetryState, stale_after_sec: float = 180.0) -> None:
        self.telemetry = telemetry
        self.stale_after_sec = stale_after_sec
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start(self, host: str, port: int) -> None:
        app = web.Application()
        app.router.add_get("/healthz", self.healthz)
        app.router.add_get("/readyz", self.readyz)
        app.router.add_get("/metrics", self.metrics)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host=host, port=port)
        await self._site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    async def healthz(self, _request: web.Request) -> web.Response:
        age = self.telemetry.update_stream_age()
        status = 200 if age <= self.stale_after_sec else 503
        return web.json_response({"ok": status == 200, "stream_age_seconds": age}, status=status)

    async def readyz(self, _request: web.Request) -> web.Response:
        age = self.telemetry.update_stream_age()
        ready = self.telemetry.ready and self.telemetry.stream_connected and age <= self.stale_after_sec
        return web.json_response({"ready": ready}, status=200 if ready else 503)

    async def metrics(self, _request: web.Request) -> web.Response:
        return web.Response(body=generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})
