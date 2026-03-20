"""
Playwright browser lifecycle management for Territorial.io bot.
"""

from __future__ import annotations

import logging
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Error as PlaywrightError
from playwright.sync_api import Page, Playwright, sync_playwright


class BrowserManager:
    """
    Manage the lifecycle of the Playwright browser, context, and active page.
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize the browser manager with configuration.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.config = config
        self.game_config = config["game"]
        self.logger = logger or logging.getLogger("browser_manager")
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    def launch(self) -> Page:
        """
        Launch a Chromium browser session and create a fresh page.

        Returns:
            Page: Active Playwright page object.

        Raises:
            PlaywrightError: If Chromium could not be launched.
        """

        if self.page is not None:
            return self.page

        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=self.game_config["headless"],
                args=[
                    "--disable-notifications",
                    "--disable-geolocation",
                    "--disable-popup-blocking",
                    "--disable-infobars",
                    "--mute-audio",
                ],
            )
            self.context = self.browser.new_context(
                viewport={
                    "width": self.game_config["window_width"],
                    "height": self.game_config["window_height"],
                },
                ignore_https_errors=True,
                service_workers="block",
            )
            self.page = self.context.new_page()
            self.page.set_default_timeout(self.game_config["browser_timeout_ms"])
            self.logger.info("Browser launched successfully")
            return self.page
        except PlaywrightError as exc:
            self.logger.exception("Failed to launch browser: %s", exc)
            self.close()
            raise

    def get_page(self) -> Page:
        """
        Return the active browser page, launching the browser if needed.

        Returns:
            Page: Active Playwright page object.
        """

        if self.page is None:
            return self.launch()
        return self.page

    def restart(self) -> Page:
        """
        Restart the browser session and return a fresh page.

        Returns:
            Page: Newly created Playwright page object.
        """

        self.logger.warning("Restarting browser session")
        self.close()
        return self.launch()

    def close(self) -> None:
        """
        Close the page, context, browser, and Playwright runtime gracefully.
        """

        for resource_name in ("page", "context", "browser", "playwright"):
            resource = getattr(self, resource_name)
            if resource is None:
                continue
            try:
                resource.close() if resource_name != "playwright" else resource.stop()
            except PlaywrightError as exc:
                self.logger.warning("Failed to close %s cleanly: %s", resource_name, exc)
            finally:
                setattr(self, resource_name, None)
