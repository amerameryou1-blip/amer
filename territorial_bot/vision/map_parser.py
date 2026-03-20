"""
Map parsing utilities for Territorial.io screenshot analysis.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vision.color_profiles import detect_color_mask


@dataclass
class MapState:
    """
    Structured map features extracted from a Territorial.io frame.
    """

    player_territory_pixels: int
    enemy_territory_pixels: int
    neutral_territory_pixels: int
    player_territory_ratio: float
    player_centroid: tuple[int, int]
    nearest_enemy_distance: float
    nearest_enemy_direction: tuple[float, float]
    nearest_neutral_distance: float
    nearest_neutral_direction: tuple[float, float]
    territory_growth_rate: float
    is_surrounded: bool


class MapParser:
    """
    Parse downsampled map screenshots into compact environment state features.
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize the map parser and its cached state.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.config = config
        self.vision_config = config["vision"]
        self.logger = logger or logging.getLogger("map_parser")
        self.previous_player_pixels: int | None = None
        self.process_width = 320
        self.process_height = 180

    def reset(self) -> None:
        """
        Reset cached frame-to-frame state for a new episode.
        """

        self.previous_player_pixels = None

    def _empty_state(self) -> MapState:
        """
        Create a zeroed map state placeholder.

        Returns:
            MapState: Empty fallback map state.
        """

        center = (self.process_width // 2, self.process_height // 2)
        return MapState(
            player_territory_pixels=0,
            enemy_territory_pixels=0,
            neutral_territory_pixels=0,
            player_territory_ratio=0.0,
            player_centroid=center,
            nearest_enemy_distance=float(math.hypot(self.process_width, self.process_height)),
            nearest_enemy_direction=(0.0, 0.0),
            nearest_neutral_distance=float(math.hypot(self.process_width, self.process_height)),
            nearest_neutral_direction=(0.0, 0.0),
            territory_growth_rate=0.0,
            is_surrounded=False,
        )

    def _crop_map_region(self, screenshot_bgr: np.ndarray) -> np.ndarray:
        """
        Crop the configured map region from the input screenshot.

        Args:
            screenshot_bgr (np.ndarray): Full-page screenshot in BGR.

        Returns:
            np.ndarray: Cropped map region.
        """

        x, y, width, height = self.vision_config["map_region"]
        frame_height, frame_width = screenshot_bgr.shape[:2]
        x1 = max(0, min(frame_width, x))
        y1 = max(0, min(frame_height, y))
        x2 = max(x1 + 1, min(frame_width, x + width))
        y2 = max(y1 + 1, min(frame_height, y + height))
        return screenshot_bgr[y1:y2, x1:x2]

    def _grayscale_fallback_mask(self, region_bgr: np.ndarray) -> np.ndarray:
        """
        Produce a coarse fallback player mask from grayscale intensity.

        Args:
            region_bgr (np.ndarray): Downsampled map region in BGR.

        Returns:
            np.ndarray: Binary fallback mask.
        """

        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    def _largest_contour(self, mask: np.ndarray) -> np.ndarray | None:
        """
        Return the largest contour in a binary mask.

        Args:
            mask (np.ndarray): Binary mask.

        Returns:
            np.ndarray | None: Largest contour, or None if no contour exists.
        """

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _compute_centroid(self, mask: np.ndarray) -> tuple[int, int]:
        """
        Compute the centroid of the player's main territory blob.

        Args:
            mask (np.ndarray): Player territory mask.

        Returns:
            tuple[int, int]: Centroid coordinates in analysis-space pixels.
        """

        contour = self._largest_contour(mask)
        if contour is None or cv2.contourArea(contour) < self.vision_config["min_territory_pixels"]:
            return self.process_width // 2, self.process_height // 2

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return self.process_width // 2, self.process_height // 2
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        return centroid_x, centroid_y

    def _compute_nearest_target(
        self,
        centroid: tuple[int, int],
        target_mask: np.ndarray,
    ) -> tuple[float, tuple[float, float]]:
        """
        Compute distance and unit direction from centroid to nearest target pixel.

        Args:
            centroid (tuple[int, int]): Player centroid.
            target_mask (np.ndarray): Target binary mask.

        Returns:
            tuple[float, tuple[float, float]]: Distance and normalized direction vector.
        """

        points = cv2.findNonZero(target_mask)
        if points is None:
            return float(math.hypot(self.process_width, self.process_height)), (0.0, 0.0)

        centroid_vec = np.array(centroid, dtype=np.float32)
        target_points = points.reshape(-1, 2).astype(np.float32)
        deltas = target_points - centroid_vec
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        min_index = int(np.argmin(distances))
        nearest_delta = deltas[min_index]
        nearest_distance = float(distances[min_index])
        if nearest_distance == 0:
            return 0.0, (0.0, 0.0)
        direction = (float(nearest_delta[0] / nearest_distance), float(nearest_delta[1] / nearest_distance))
        return nearest_distance, direction

    def _compute_is_surrounded(
        self,
        player_mask: np.ndarray,
        enemy_mask: np.ndarray,
        neutral_mask: np.ndarray,
        background_mask: np.ndarray,
        ui_mask: np.ndarray,
    ) -> bool:
        """
        Estimate whether the player territory is mostly enclosed by enemies.

        Args:
            player_mask (np.ndarray): Player territory mask.
            enemy_mask (np.ndarray): Enemy territory mask.
            neutral_mask (np.ndarray): Neutral territory mask.
            background_mask (np.ndarray): Background mask.
            ui_mask (np.ndarray): UI-element mask.

        Returns:
            bool: True when the surrounding ring is enemy-dominated.
        """

        contour = self._largest_contour(player_mask)
        if contour is None:
            return False

        territory = np.zeros_like(player_mask)
        cv2.drawContours(territory, [contour], -1, 255, thickness=-1)
        kernel = np.ones((7, 7), dtype=np.uint8)
        expanded = cv2.dilate(territory, kernel, iterations=1)
        ring = cv2.subtract(expanded, territory)
        valid_ring = cv2.bitwise_and(ring, cv2.bitwise_not(background_mask))
        valid_ring = cv2.bitwise_and(valid_ring, cv2.bitwise_not(ui_mask))
        valid_pixels = int(cv2.countNonZero(valid_ring))
        if valid_pixels == 0:
            return False

        enemy_pixels = int(cv2.countNonZero(cv2.bitwise_and(enemy_mask, valid_ring)))
        neutral_pixels = int(cv2.countNonZero(cv2.bitwise_and(neutral_mask, valid_ring)))
        enemy_ratio = enemy_pixels / max(valid_pixels, 1)
        return enemy_ratio >= 0.65 and enemy_pixels > neutral_pixels

    def parse(self, screenshot_bgr: np.ndarray | None) -> MapState:
        """
        Parse a screenshot into a structured map state.

        Args:
            screenshot_bgr (np.ndarray | None): Full-page screenshot in BGR format.

        Returns:
            MapState: Parsed state features.
        """

        if screenshot_bgr is None or screenshot_bgr.size == 0:
            self.logger.warning("Received empty screenshot during parse")
            return self._empty_state()

        map_crop = self._crop_map_region(screenshot_bgr)
        resized = cv2.resize(map_crop, (self.process_width, self.process_height), interpolation=cv2.INTER_AREA)

        tolerance = self.vision_config["color_tolerance"]
        ui_mask = detect_color_mask(resized, "ui_elements", tolerance=tolerance)
        background_mask = detect_color_mask(resized, "background", tolerance=tolerance)
        player_mask = detect_color_mask(resized, "player_territory", tolerance=tolerance)
        enemy_mask = detect_color_mask(resized, "enemy_territory", tolerance=tolerance)
        neutral_mask = detect_color_mask(resized, "neutral_territory", tolerance=tolerance)

        player_mask = cv2.bitwise_and(player_mask, cv2.bitwise_not(ui_mask))
        enemy_mask = cv2.bitwise_and(enemy_mask, cv2.bitwise_not(ui_mask))
        neutral_mask = cv2.bitwise_and(neutral_mask, cv2.bitwise_not(ui_mask))

        if (
            self.vision_config["use_grayscale_fallback"]
            and cv2.countNonZero(player_mask) < self.vision_config["min_territory_pixels"]
        ):
            player_mask = self._grayscale_fallback_mask(resized)

        valid_map = cv2.bitwise_not(cv2.bitwise_or(background_mask, ui_mask))
        player_pixels = int(cv2.countNonZero(cv2.bitwise_and(player_mask, valid_map)))
        enemy_pixels = int(cv2.countNonZero(cv2.bitwise_and(enemy_mask, valid_map)))
        neutral_pixels = int(cv2.countNonZero(cv2.bitwise_and(neutral_mask, valid_map)))
        total_pixels = int(cv2.countNonZero(valid_map))
        player_ratio = player_pixels / max(total_pixels, 1)

        centroid = self._compute_centroid(player_mask)
        nearest_enemy_distance, nearest_enemy_direction = self._compute_nearest_target(centroid, enemy_mask)
        nearest_neutral_distance, nearest_neutral_direction = self._compute_nearest_target(centroid, neutral_mask)

        if self.previous_player_pixels is None:
            growth_rate = 0.0
        else:
            growth_rate = (player_pixels - self.previous_player_pixels) / max(self.previous_player_pixels, 1)

        surrounded = self._compute_is_surrounded(player_mask, enemy_mask, neutral_mask, background_mask, ui_mask)
        self.previous_player_pixels = player_pixels

        del map_crop
        del resized

        return MapState(
            player_territory_pixels=player_pixels,
            enemy_territory_pixels=enemy_pixels,
            neutral_territory_pixels=neutral_pixels,
            player_territory_ratio=player_ratio,
            player_centroid=centroid,
            nearest_enemy_distance=nearest_enemy_distance,
            nearest_enemy_direction=nearest_enemy_direction,
            nearest_neutral_distance=nearest_neutral_distance,
            nearest_neutral_direction=nearest_neutral_direction,
            territory_growth_rate=growth_rate,
            is_surrounded=surrounded,
        )
