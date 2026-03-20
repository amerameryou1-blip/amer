"""
Navigation and game-state handling for Territorial.io.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator, Page


class GameLauncher:
    """
    Navigate to Territorial.io, enter the lobby, and detect match state transitions.
    """

    def __init__(
        self,
        browser_manager: Any,
        config: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the launcher with browser manager and configuration.

        Args:
            browser_manager (Any): Browser manager instance.
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.browser_manager = browser_manager
        self.config = config
        self.game_config = config["game"]
        self.logger = logger or logging.getLogger("game_launcher")
        self.name_selectors = [
            "input[type='text']",
            "input[placeholder*='name' i]",
            "input[maxlength]",
        ]
        self.start_selectors = [
            "text=/^play$/i",
            "text=/start/i",
            "text=/new game/i",
            "button:has-text('Play')",
            "button:has-text('Start')",
        ]
        self.retry_selectors = [
            "text=/play again/i",
            "text=/continue/i",
            "text=/respawn/i",
            "text=/new game/i",
            "button:has-text('Continue')",
        ]
        self.game_over_selectors = [
            "text=/game over/i",
            "text=/defeat/i",
            "text=/defeated/i",
            "text=/play again/i",
            "text=/continue/i",
            "text=/respawn/i",
        ]
        self.victory_selectors = [
            "text=/victory/i",
            "text=/winner/i",
            "text=/you won/i",
            "text='#1'",
        ]

    def _get_page(self) -> Page:
        """
        Retrieve the active page from the browser manager.

        Returns:
            Page: Active Playwright page.
        """

        return self.browser_manager.get_page()

    def _selector_visible(self, page: Page, selector: str) -> bool:
        """
        Check whether a locator matching a selector is visible.

        Args:
            page (Page): Active Playwright page.
            selector (str): Playwright selector string.

        Returns:
            bool: True when a matching visible element exists.
        """

        try:
            locator = page.locator(selector)
            if locator.count() == 0:
                return False
            return locator.first.is_visible()
        except PlaywrightError:
            return False

    def _click_first_visible(self, page: Page, selectors: list[str]) -> bool:
        """
        Click the first visible locator from a list of selectors.

        Args:
            page (Page): Active Playwright page.
            selectors (list[str]): Candidate selectors.

        Returns:
            bool: True when a click was successfully performed.
        """

        for selector in selectors:
            try:
                locator: Locator = page.locator(selector)
                if locator.count() == 0 or not locator.first.is_visible():
                    continue
                locator.first.click()
                return True
            except PlaywrightError:
                continue
        return False

    def navigate_to_game(self) -> None:
        """
        Open the game URL and wait for the page to become ready.
        """

        page = self._get_page()
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                page.goto(self.game_config["url"], wait_until="domcontentloaded")
                page.wait_for_load_state("networkidle", timeout=self.game_config["browser_timeout_ms"])
                self.logger.info("Navigated to %s", self.game_config["url"])
                return
            except PlaywrightError as exc:
                self.logger.warning("Navigation attempt %s/%s failed: %s", attempt, retries, exc)
                time.sleep(1.0)
        raise PlaywrightError(f"Failed to navigate to {self.game_config['url']} after {retries} attempts")

    def enter_name(self, name: str) -> bool:
        """
        Enter the configured player name into the menu input field.

        Args:
            name (str): Player name to type.

        Returns:
            bool: True when a name field was found and filled.
        """

        page = self._get_page()
        for selector in self.name_selectors:
            try:
                locator = page.locator(selector)
                if locator.count() == 0 or not locator.first.is_visible():
                    continue
                locator.first.click()
                locator.first.fill("")
                locator.first.type(name, delay=35)
                self.logger.info("Entered player name '%s'", name)
                return True
            except PlaywrightError:
                continue

        try:
            filled = page.evaluate(
                """
                (value) => {
                    const candidates = Array.from(document.querySelectorAll('input'));
                    const target = candidates.find((input) => input.type === 'text' || !input.type);
                    if (!target) {
                        return false;
                    }
                    target.focus();
                    target.value = value;
                    target.dispatchEvent(new Event('input', { bubbles: true }));
                    target.dispatchEvent(new Event('change', { bubbles: true }));
                    return true;
                }
                """,
                name,
            )
            if filled:
                self.logger.info("Entered player name using JavaScript fallback")
                return True
        except PlaywrightError as exc:
            self.logger.warning("Name entry fallback failed: %s", exc)

        self.logger.warning("Could not find player name input")
        return False

    def start_match(self) -> bool:
        """
        Attempt to start a new game from the menu.

        Returns:
            bool: True when the game appears to have entered active play.
        """

        page = self._get_page()
        if self.detect_in_game():
            return True

        for attempt in range(1, 6):
            try:
                self.enter_name(self.game_config["player_name"])
                if not self._click_first_visible(page, self.start_selectors):
                    canvas = page.locator("canvas")
                    if canvas.count() > 0:
                        box = canvas.first.bounding_box()
                        if box:
                            page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.8)
                time.sleep(1.0)
                if self.detect_in_game():
                    self.logger.info("Match started successfully")
                    return True
            except PlaywrightError as exc:
                self.logger.warning("Start match attempt %s failed: %s", attempt, exc)
                time.sleep(1.0)

        self.logger.warning("Unable to confirm game start after retries")
        return False

    def detect_game_over(self) -> bool:
        """
        Detect whether the game-over or post-match screen is visible.

        Returns:
            bool: True if a game-over state is detected.
        """

        page = self._get_page()
        try:
            return any(self._selector_visible(page, selector) for selector in self.game_over_selectors)
        except Exception:
            return False

    def detect_victory(self) -> bool:
        """
        Detect whether the visible post-match state suggests a win.

        Returns:
            bool: True when a victory indicator is visible.
        """

        page = self._get_page()
        try:
            return any(self._selector_visible(page, selector) for selector in self.victory_selectors)
        except Exception:
            return False

    def detect_in_game(self) -> bool:
        """
        Determine whether the bot is currently inside an active match.

        Returns:
            bool: True when active game heuristics pass.
        """

        page = self._get_page()
        try:
            if self.detect_game_over():
                return False

            canvas = page.locator("canvas")
            canvas_visible = canvas.count() > 0 and canvas.first.is_visible()
            menu_visible = any(self._selector_visible(page, selector) for selector in self.start_selectors)
            name_field_visible = any(self._selector_visible(page, selector) for selector in self.name_selectors)
            return canvas_visible and not menu_visible and not name_field_visible
        except PlaywrightError:
            return False

    def handle_menu_state(self) -> None:
        """
        Recover the browser into a clean pre-game state ready to start a match.
        """

        page = self._get_page()
        if not page.url or "territorial.io" not in page.url:
            self.navigate_to_game()

        if self.detect_game_over():
            self.logger.info("Game-over screen detected; attempting to return to menu")
            if not self._click_first_visible(page, self.retry_selectors):
                try:
                    page.keyboard.press("Enter")
                except PlaywrightError:
                    pass
            time.sleep(1.5)

        if self.detect_in_game():
            return

        self.navigate_to_game()
        self.enter_name(self.game_config["player_name"])
