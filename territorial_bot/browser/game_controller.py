"""
Mouse and keyboard controls for Territorial.io.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page


class GameController:
    """
    Execute the Territorial.io action space with real browser inputs.
    """

    def __init__(self, page: Page, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize the controller.

        Args:
            page (Page): Active Playwright page.
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.page = page
        self.config = config
        self.actions_config = config["actions"]
        self.game_config = config["game"]
        self.logger = logger or logging.getLogger("game_controller")

    def set_page(self, page: Page) -> None:
        """
        Update the page after a browser restart.

        Args:
            page (Page): Fresh Playwright page.
        """

        self.page = page

    def _humanize_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """
        Clamp coordinates to the viewport and add a small human-like offset.

        Args:
            x (int): Target x coordinate.
            y (int): Target y coordinate.

        Returns:
            tuple[int, int]: Adjusted coordinates.
        """

        adjusted_x = max(0, min(self.game_config["window_width"] - 1, x))
        adjusted_y = max(0, min(self.game_config["window_height"] - 1, y))
        return adjusted_x, adjusted_y

    def _post_action_delay(self) -> None:
        """
        Sleep for the configured action delay.
        """

        time.sleep(self.actions_config["action_delay_ms"] / 1000.0)

    def click(self, x: int, y: int) -> None:
        """
        Perform a click at the given screen coordinates.

        Args:
            x (int): Screen x coordinate.
            y (int): Screen y coordinate.
        """

        target_x, target_y = self._humanize_coordinates(x, y)
        self.page.mouse.click(target_x, target_y)
        self._post_action_delay()

    def click_and_hold(self, x: int, y: int, duration_ms: int) -> None:
        """
        Perform a click-and-hold interaction.

        Args:
            x (int): Screen x coordinate.
            y (int): Screen y coordinate.
            duration_ms (int): Hold duration in milliseconds.
        """

        target_x, target_y = self._humanize_coordinates(x, y)
        self.page.mouse.move(target_x, target_y)
        self.page.mouse.down()
        time.sleep(duration_ms / 1000.0)
        self.page.mouse.up()
        self._post_action_delay()

    def move_mouse(self, x: int, y: int) -> None:
        """
        Move the mouse cursor to a screen position.

        Args:
            x (int): Screen x coordinate.
            y (int): Screen y coordinate.
        """

        target_x, target_y = self._humanize_coordinates(x, y)
        self.page.mouse.move(target_x, target_y)
        self._post_action_delay()

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Drag the mouse between two points.

        Args:
            x1 (int): Start x coordinate.
            y1 (int): Start y coordinate.
            x2 (int): End x coordinate.
            y2 (int): End y coordinate.
        """

        start_x, start_y = self._humanize_coordinates(x1, y1)
        end_x, end_y = self._humanize_coordinates(x2, y2)
        self.page.mouse.move(start_x, start_y)
        self.page.mouse.down()
        self.page.mouse.move(end_x, end_y, steps=8)
        self.page.mouse.up()
        self._post_action_delay()

    def press_key(self, key: str) -> None:
        """
        Press a keyboard key on the game canvas.

        Args:
            key (str): Playwright key name.
        """

        self.page.keyboard.press(key)
        self._post_action_delay()

    def attack_player_under_mouse(self) -> None:
        """
        Press SPACE to attack the player currently under the mouse cursor.
        """

        self.page.keyboard.press("Space")
        self._post_action_delay()

    def move_and_attack(self, target_x: int, target_y: int) -> tuple[int, int]:
        """
        Move the mouse to a target point and trigger a spacebar attack.

        Args:
            target_x (int): Target x coordinate.
            target_y (int): Target y coordinate.

        Returns:
            tuple[int, int]: Final target coordinates.
        """

        final_x, final_y = self._humanize_coordinates(target_x, target_y)
        self.page.mouse.move(final_x, final_y)
        time.sleep(0.1)
        self.page.keyboard.press("Space")
        self._post_action_delay()
        return final_x, final_y

    def execute_action(self, action_id: int, map_state: Any | None = None) -> tuple[int, int]:
        """
        Execute one action from the 73-action Territorial.io action space.

        Args:
            action_id (int): Action index.
            map_state (Any | None): Optional parsed map state for centroid-based actions.

        Returns:
            tuple[int, int]: Coordinates used for the action, or `(-1, -1)` for space-only attack.
        """

        try:
            if 0 <= action_id <= 63:
                grid_size = int(self.actions_config["grid_size"])
                cell_x = action_id % grid_size
                cell_y = action_id // grid_size
                pixel_x = (
                    int(self.actions_config["playable_x_start"])
                    + cell_x * 133
                    + 66
                    + random.randint(-10, 10)
                )
                pixel_y = (
                    int(self.actions_config["playable_y_start"])
                    + cell_y * 80
                    + 40
                    + random.randint(-5, 5)
                )
                final_x, final_y = self._humanize_coordinates(pixel_x, pixel_y)
                self.page.mouse.click(final_x, final_y)
                self._post_action_delay()
                return final_x, final_y

            if 64 <= action_id <= 71:
                compass_offsets = [
                    (0, -1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                    (-1, 1),
                    (-1, 0),
                    (-1, -1),
                ]
                direction_index = action_id - 64
                base_x = self.game_config["window_width"] // 2
                base_y = self.game_config["window_height"] // 2
                if map_state is not None and getattr(map_state, "player_centroid", None) is not None:
                    base_x, base_y = map_state.player_centroid
                offset_x, offset_y = compass_offsets[direction_index]
                distance = int(self.actions_config["directional_offset_px"])
                target_x = int(base_x + offset_x * distance)
                target_y = int(base_y + offset_y * distance)
                return self.move_and_attack(target_x, target_y)

            if action_id == 72:
                self.attack_player_under_mouse()
                return -1, -1

            raise ValueError(f"Unsupported action id: {action_id}")
        except PlaywrightError as exc:
            self.logger.exception("Failed to execute action %s: %s", action_id, exc)
            raise
