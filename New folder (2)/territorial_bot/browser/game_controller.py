"""
Mouse and keyboard controls for Territorial.io bot actions.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

from agent.action_space import ActionSpace


class GameController:
    """
    Execute browser input actions that correspond to reinforcement learning actions.
    """

    def __init__(self, page: Page, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize the controller with a Playwright page and configuration.

        Args:
            page (Page): Active Playwright page object.
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.page = page
        self.config = config
        self.actions_config = config["actions"]
        self.game_config = config["game"]
        self.action_space = ActionSpace(config)
        self.logger = logger or logging.getLogger("game_controller")
        self.player_centroid: tuple[int, int] | None = None

    def set_page(self, page: Page) -> None:
        """
        Update the active Playwright page after a browser restart.

        Args:
            page (Page): Fresh Playwright page object.
        """

        self.page = page

    def set_player_centroid(self, centroid: tuple[int, int] | None) -> None:
        """
        Store the latest observed player centroid for directional actions.

        Args:
            centroid (tuple[int, int] | None): Latest player centroid in analysis-space pixels.
        """

        self.player_centroid = centroid

    def _humanize_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """
        Add a small random offset and clamp coordinates to the viewport.

        Args:
            x (int): Target x coordinate.
            y (int): Target y coordinate.

        Returns:
            tuple[int, int]: Humanized and clamped coordinate pair.
        """

        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        width = self.game_config["window_width"]
        height = self.game_config["window_height"]
        human_x = max(0, min(width - 1, x + offset_x))
        human_y = max(0, min(height - 1, y + offset_y))
        return human_x, human_y

    def _post_action_delay(self) -> None:
        """
        Sleep for the configured post-action delay.
        """

        time.sleep(self.actions_config["action_delay_ms"] / 1000.0)

    def click(self, x: int, y: int) -> None:
        """
        Perform a humanized mouse click at the requested position.

        Args:
            x (int): Screen x coordinate.
            y (int): Screen y coordinate.
        """

        target_x, target_y = self._humanize_coordinates(x, y)
        try:
            self.page.mouse.move(target_x, target_y)
            self.page.mouse.click(target_x, target_y)
            self._post_action_delay()
        except PlaywrightError as exc:
            self.logger.exception("Mouse click failed at (%s, %s): %s", target_x, target_y, exc)
            raise

    def click_and_hold(self, x: int, y: int, duration_ms: int) -> None:
        """
        Press and hold the mouse button at the requested position.

        Args:
            x (int): Screen x coordinate.
            y (int): Screen y coordinate.
            duration_ms (int): Hold duration in milliseconds.
        """

        target_x, target_y = self._humanize_coordinates(x, y)
        try:
            self.page.mouse.move(target_x, target_y)
            self.page.mouse.down()
            time.sleep(duration_ms / 1000.0)
            self.page.mouse.up()
            self._post_action_delay()
        except PlaywrightError as exc:
            self.logger.exception("Click-and-hold failed at (%s, %s): %s", target_x, target_y, exc)
            raise

    def move_mouse(self, x: int, y: int) -> None:
        """
        Move the mouse cursor to the requested position without clicking.

        Args:
            x (int): Screen x coordinate.
            y (int): Screen y coordinate.
        """

        target_x, target_y = self._humanize_coordinates(x, y)
        try:
            self.page.mouse.move(target_x, target_y)
            self._post_action_delay()
        except PlaywrightError as exc:
            self.logger.exception("Mouse move failed to (%s, %s): %s", target_x, target_y, exc)
            raise

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Click and drag between two points.

        Args:
            x1 (int): Starting x coordinate.
            y1 (int): Starting y coordinate.
            x2 (int): Ending x coordinate.
            y2 (int): Ending y coordinate.
        """

        start_x, start_y = self._humanize_coordinates(x1, y1)
        end_x, end_y = self._humanize_coordinates(x2, y2)
        try:
            self.page.mouse.move(start_x, start_y)
            self.page.mouse.down()
            self.page.mouse.move(end_x, end_y, steps=6)
            self.page.mouse.up()
            self._post_action_delay()
        except PlaywrightError as exc:
            self.logger.exception(
                "Mouse drag failed from (%s, %s) to (%s, %s): %s",
                start_x,
                start_y,
                end_x,
                end_y,
                exc,
            )
            raise

    def press_key(self, key: str) -> None:
        """
        Press a keyboard key on the active page.

        Args:
            key (str): Key name understood by Playwright.
        """

        try:
            self.page.keyboard.press(key)
            self._post_action_delay()
        except PlaywrightError as exc:
            self.logger.exception("Key press failed for '%s': %s", key, exc)
            raise

    def execute_action(self, action_id: int) -> tuple[int, int]:
        """
        Map an action ID to a concrete browser interaction and execute it.

        Args:
            action_id (int): Action index from the agent action space.

        Returns:
            tuple[int, int]: Screen coordinates used for the action.
        """

        target_x, target_y = self.action_space.get_action_coordinates(action_id, self.player_centroid)
        self.click_and_hold(target_x, target_y, self.actions_config["hold_duration_ms"])
        return target_x, target_y
