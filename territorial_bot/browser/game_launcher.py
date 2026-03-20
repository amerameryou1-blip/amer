"""
Canvas-aware launcher and game-state detection for Territorial.io.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page


class GameLauncher:
    """
    Control the menu flow and detect game states from real screenshots.
    """

    def __init__(self, page: Page, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize the launcher.

        Args:
            page (Page): Active Playwright page.
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.page = page
        self.config = config
        self.game_config = config["game"]
        self.vision_config = config["vision"]
        self.logger = logger or logging.getLogger("launcher")

    def set_page(self, page: Page) -> None:
        """
        Update the active page after a browser restart.

        Args:
            page (Page): Fresh Playwright page.
        """

        self.page = page

    def _capture_bgr(self) -> np.ndarray | None:
        """
        Capture the current page as a BGR image.

        Returns:
            np.ndarray | None: Screenshot in BGR format, or None on failure.
        """

        try:
            image_bytes = self.page.screenshot(type="jpeg", quality=70)
        except PlaywrightError as exc:
            self.logger.warning("Launcher screenshot capture failed: %s", exc)
            return None

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    def _crop(self, screenshot_bgr: np.ndarray, region: list[int]) -> np.ndarray:
        """
        Crop a region from a screenshot while staying in bounds.

        Args:
            screenshot_bgr (np.ndarray): Source screenshot.
            region (list[int]): `[x, y, width, height]` crop region.

        Returns:
            np.ndarray: Cropped image region.
        """

        x, y, width, height = region
        image_height, image_width = screenshot_bgr.shape[:2]
        x1 = max(0, min(image_width, x))
        y1 = max(0, min(image_height, y))
        x2 = max(x1 + 1, min(image_width, x + width))
        y2 = max(y1 + 1, min(image_height, y + height))
        return screenshot_bgr[y1:y2, x1:x2]

    def _region_dark_ratio(self, screenshot_bgr: np.ndarray, region: list[int], value_threshold: int = 80) -> float:
        """
        Measure the fraction of dark pixels inside a region.

        Args:
            screenshot_bgr (np.ndarray): Source screenshot.
            region (list[int]): Region to inspect.
            value_threshold (int): HSV value cutoff for darkness.

        Returns:
            float: Ratio of dark pixels in the cropped region.
        """

        crop = self._crop(screenshot_bgr, region)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2] < value_threshold))

    def navigate_to_game(self) -> bool:
        """
        Navigate to Territorial.io and wait for the main menu canvas to appear.

        Returns:
            bool: True when navigation succeeded.
        """

        try:
            self.page.goto(self.game_config["url"])
            self.page.wait_for_load_state("networkidle", timeout=self.game_config["browser_timeout_ms"])
            time.sleep(2.0)
            page_title = self.page.title()
            success = "territorial" in self.page.url.lower() or "territorial" in page_title.lower()
            self.logger.info("Navigated to game | url=%s | title=%s | success=%s", self.page.url, page_title, success)
            return success
        except PlaywrightError as exc:
            self.logger.exception("Failed to navigate to Territorial.io: %s", exc)
            return False

    def set_player_name(self, name: str) -> bool:
        """
        Clear and set the player name field on the main menu.

        Args:
            name (str): Desired in-game player name.

        Returns:
            bool: True when the name was set successfully.
        """

        selectors = ['input[type="text"]', "input"]
        for selector in selectors:
            try:
                locator = self.page.locator(selector)
                if locator.count() == 0:
                    continue
                target = locator.first
                target.click(click_count=3)
                target.fill("")
                target.type(name, delay=30)
                self.logger.info("Player name set via selector '%s' to '%s'", selector, name)
                return True
            except PlaywrightError:
                continue

        try:
            self.page.mouse.click(726, 377, click_count=3)
            self.page.keyboard.press("Control+A")
            self.page.keyboard.press("Backspace")
            self.page.keyboard.type(name, delay=30)
            self.logger.info("Player name set using coordinate fallback to '%s'", name)
            return True
        except PlaywrightError as exc:
            self.logger.exception("Failed to set player name '%s': %s", name, exc)
            return False

    def click_multiplayer(self) -> bool:
        """
        Click the Multiplayer menu button and wait for matchmaking to begin.

        Returns:
            bool: True when the click was issued successfully.
        """

        try:
            self.page.click("text=Multiplayer", timeout=1500)
            time.sleep(3.0)
            self.logger.info("Clicked Multiplayer via text selector")
            return True
        except PlaywrightError:
            pass

        try:
            self.page.mouse.click(672, 452)
            time.sleep(3.0)
            self.logger.info("Clicked Multiplayer via coordinate fallback")
            return True
        except PlaywrightError as exc:
            self.logger.exception("Failed to click Multiplayer: %s", exc)
            return False

    def wait_for_game_start(self) -> bool:
        """
        Wait until the leaderboard region indicates that a live match has started.

        Returns:
            bool: True if active gameplay was detected before timeout.
        """

        start_time = time.time()
        timeout_seconds = 10.0
        leaderboard_region = self.vision_config["leaderboard_region"]
        while (time.time() - start_time) < timeout_seconds:
            screenshot_bgr = self._capture_bgr()
            if screenshot_bgr is None:
                time.sleep(0.5)
                continue
            dark_ratio = self._region_dark_ratio(screenshot_bgr, leaderboard_region)
            if dark_ratio > 0.45:
                self.logger.info("Detected game start from leaderboard dark region | dark_ratio=%.3f", dark_ratio)
                return True
            time.sleep(0.5)
        self.logger.warning("Timed out waiting for game start")
        return False

    def detect_defeat(self, screenshot_bgr: np.ndarray) -> bool:
        """
        Detect the DEFEAT popup from the center-screen dark-green rectangle.

        Args:
            screenshot_bgr (np.ndarray): Full screenshot in BGR format.

        Returns:
            bool: True if defeat popup heuristics matched.
        """

        defeat_region = self.vision_config["defeat_check_region"]
        crop = self._crop(screenshot_bgr, defeat_region)
        if crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        dark_green_mask = cv2.inRange(hsv, np.array([60, 50, 30], dtype=np.uint8), np.array([90, 150, 80], dtype=np.uint8))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200], dtype=np.uint8), np.array([179, 60, 255], dtype=np.uint8))
        dark_green_ratio = cv2.countNonZero(dark_green_mask) / float(dark_green_mask.size)
        white_ratio = cv2.countNonZero(white_mask) / float(white_mask.size)
        detected = dark_green_ratio > 0.30 or (dark_green_ratio > 0.20 and white_ratio > 0.01)
        if detected:
            self.logger.info(
                "Defeat popup detected | dark_green_ratio=%.3f | white_ratio=%.3f",
                dark_green_ratio,
                white_ratio,
            )
        return detected

    def detect_in_game(self, screenshot_bgr: np.ndarray) -> bool:
        """
        Detect active in-game state from leaderboard and troop-bar presence.

        Args:
            screenshot_bgr (np.ndarray): Full screenshot in BGR format.

        Returns:
            bool: True if gameplay HUD is visible.
        """

        leaderboard_dark = self._region_dark_ratio(screenshot_bgr, [0, 0, 310, 80]) > 0.45
        troop_crop = self._crop(screenshot_bgr, [450, 730, 630, 38])
        if troop_crop.size == 0:
            return False
        troop_hsv = cv2.cvtColor(troop_crop, cv2.COLOR_BGR2HSV)
        troop_colored = float(np.mean((troop_hsv[:, :, 1] > 60) & (troop_hsv[:, :, 2] > 60))) > 0.25
        return leaderboard_dark and troop_colored

    def detect_main_menu(self, screenshot_bgr: np.ndarray) -> bool:
        """
        Detect the main menu from HTML overlay visibility or bright center buttons.

        Args:
            screenshot_bgr (np.ndarray): Full screenshot in BGR format.

        Returns:
            bool: True if main-menu heuristics matched.
        """

        try:
            locator = self.page.query_selector("text=Multiplayer")
            if locator is not None:
                self.logger.info("Main menu detected via Multiplayer selector")
                return True
        except PlaywrightError:
            pass

        center_crop = self._crop(screenshot_bgr, [460, 360, 500, 220])
        if center_crop.size == 0:
            return False
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        bright_button_ratio = float(np.mean((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 90)))
        return bright_button_ratio > 0.20

    def close_defeat_popup(self) -> bool:
        """
        Close the defeat popup and return to the main menu.

        Returns:
            bool: True if the click sequence completed.
        """

        try:
            self.page.mouse.click(1045, 617)
            time.sleep(0.5)
            self.page.mouse.click(30, 748)
            time.sleep(1.0)
            self.logger.info("Closed defeat popup and clicked exit")
            return True
        except PlaywrightError as exc:
            self.logger.exception("Failed to close defeat popup: %s", exc)
            return False

    def handle_state(self, screenshot_bgr: np.ndarray) -> str:
        """
        Detect the current high-level game state from a screenshot.

        Args:
            screenshot_bgr (np.ndarray): Full screenshot in BGR format.

        Returns:
            str: One of `menu`, `in_game`, `defeat`, or `unknown`.
        """

        if self.detect_defeat(screenshot_bgr):
            self.logger.info("Launcher state detected: defeat")
            return "defeat"
        if self.detect_in_game(screenshot_bgr):
            self.logger.info("Launcher state detected: in_game")
            return "in_game"
        if self.detect_main_menu(screenshot_bgr):
            self.logger.info("Launcher state detected: menu")
            return "menu"
        self.logger.info("Launcher state detected: unknown")
        return "unknown"
