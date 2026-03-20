"""
Fast screenshot capture utilities for Territorial.io bot.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

from utils.file_utils import ensure_dir


class ScreenshotCapture:
    """
    Capture Playwright screenshots and convert them into OpenCV BGR arrays.
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize screenshot capture configuration.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.config = config
        self.log_dir = Path(config["training"]["log_dir"])
        self.logger = logger or logging.getLogger("screenshot")
        self.debug_dir = self.log_dir / "debug_frames"

    def _bytes_to_bgr(self, raw_bytes: bytes) -> np.ndarray:
        """
        Convert JPEG bytes into an OpenCV BGR array.

        Args:
            raw_bytes (bytes): JPEG-encoded screenshot bytes.

        Returns:
            np.ndarray: Screenshot in BGR color space.
        """

        with Image.open(io.BytesIO(raw_bytes)) as image:
            rgb_image = image.convert("RGB")
            rgb_array = np.array(rgb_image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def capture(self, page: Page) -> np.ndarray | None:
        """
        Capture a full-page screenshot as a BGR image.

        Args:
            page (Page): Active Playwright page.

        Returns:
            np.ndarray | None: Screenshot in BGR format, or None on failure.
        """

        try:
            raw_bytes = page.screenshot(type="jpeg", quality=70)
            return self._bytes_to_bgr(raw_bytes)
        except PlaywrightError as exc:
            self.logger.warning("Full screenshot capture failed: %s", exc)
            return None

    def capture_region(self, page: Page, x: int, y: int, w: int, h: int) -> np.ndarray | None:
        """
        Capture a cropped region of the page as a BGR image.

        Args:
            page (Page): Active Playwright page.
            x (int): Clip origin x.
            y (int): Clip origin y.
            w (int): Clip width.
            h (int): Clip height.

        Returns:
            np.ndarray | None: Cropped screenshot in BGR format, or None on failure.
        """

        try:
            raw_bytes = page.screenshot(
                type="jpeg",
                quality=70,
                clip={"x": x, "y": y, "width": w, "height": h},
            )
            return self._bytes_to_bgr(raw_bytes)
        except PlaywrightError as exc:
            self.logger.warning("Regional screenshot capture failed: %s", exc)
            return None

    def save_debug_frame(self, img: np.ndarray, episode: int, step: int) -> Path:
        """
        Save a debug screenshot frame to disk.

        Args:
            img (np.ndarray): BGR image to write.
            episode (int): Episode index.
            step (int): Step index.

        Returns:
            Path: Saved image path.
        """

        ensure_dir(self.debug_dir)
        output_path = self.debug_dir / f"episode_{episode:05d}_step_{step:04d}.jpg"
        cv2.imwrite(str(output_path), img)
        return output_path
