"""
Map parser grounded in Territorial.io canvas screenshots.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vision.color_profiles import COLOR_PROFILES, detect_player_color_from_troopbar, get_hud_mask


@dataclass
class MapState:
    """
    Parsed gameplay state derived from a screenshot.
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
    idle_steps: int
    player_color_hsv: tuple[int, int, int] | None
    relative_rank_ratio: float


class MapParser:
    """
    Parse the Territorial.io canvas into player, neutral, water, and enemy regions.
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize the parser and its per-episode state.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.config = config
        self.vision_config = config["vision"]
        self.logger = logger or logging.getLogger("map_parser")
        self.player_color_lower: np.ndarray | None = None
        self.player_color_upper: np.ndarray | None = None
        self.player_color_hsv: tuple[int, int, int] | None = None
        self.prev_player_pixels = 0
        self.idle_counter = 0
        self.PROCESS_WIDTH = int(self.vision_config["process_width"])
        self.PROCESS_HEIGHT = int(self.vision_config["process_height"])
        self.last_masks: dict[str, np.ndarray] = {}
        self.last_original_shape: tuple[int, int] | None = None

    def reset(self) -> None:
        """
        Reset cached episode state, including tracked player color.
        """

        self.player_color_lower = None
        self.player_color_upper = None
        self.player_color_hsv = None
        self.prev_player_pixels = 0
        self.idle_counter = 0
        self.last_masks = {}
        self.last_original_shape = None

    def detect_and_set_player_color(self, screenshot_bgr: np.ndarray) -> bool:
        """
        Detect the player's current territory color from the troop bar.

        Args:
            screenshot_bgr (np.ndarray): Full-resolution screenshot.

        Returns:
            bool: True when the player color was detected successfully.
        """

        sample_x, sample_y = self.vision_config["troop_bar_sample_pixel"]
        self.logger.info("Sampling troop bar for player color at pixel (%s, %s)", sample_x, sample_y)
        result = detect_player_color_from_troopbar(
            screenshot_bgr,
            sample_pixel=tuple(self.vision_config["troop_bar_sample_pixel"]),
        )
        if result is None:
            self.logger.warning("Could not detect player color from troop bar sample")
            return False

        lower, upper, median_hsv = result
        self.player_color_lower = lower
        self.player_color_upper = upper
        self.player_color_hsv = median_hsv
        self.logger.info(
            "Player color detected successfully | hsv=%s | lower=%s | upper=%s",
            self.player_color_hsv,
            self.player_color_lower.tolist(),
            self.player_color_upper.tolist(),
        )
        return True

    def _empty_state(self) -> MapState:
        """
        Return an empty fallback state when parsing fails.

        Returns:
            MapState: Zeroed fallback map state.
        """

        center = (self.config["game"]["window_width"] // 2, self.config["game"]["window_height"] // 2)
        return MapState(
            player_territory_pixels=0,
            enemy_territory_pixels=0,
            neutral_territory_pixels=0,
            player_territory_ratio=0.0,
            player_centroid=center,
            nearest_enemy_distance=999.0,
            nearest_enemy_direction=(0.0, 0.0),
            nearest_neutral_distance=999.0,
            nearest_neutral_direction=(0.0, 0.0),
            territory_growth_rate=0.0,
            is_surrounded=False,
            idle_steps=self.idle_counter,
            player_color_hsv=self.player_color_hsv,
            relative_rank_ratio=0.0,
        )

    def _preprocess(self, screenshot_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resize, HUD-mask, and convert a screenshot for fast vision analysis.

        Args:
            screenshot_bgr (np.ndarray): Full-resolution screenshot.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Small BGR frame, small HSV frame, and HUD-valid mask.
        """

        bgr_small = cv2.resize(
            screenshot_bgr,
            (self.PROCESS_WIDTH, self.PROCESS_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        hud_mask_small = get_hud_mask(bgr_small.shape)
        hsv_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
        hsv_small[hud_mask_small == 0] = 0
        return bgr_small, hsv_small, hud_mask_small

    def _in_range(self, hsv_small: np.ndarray, lower: list[int] | np.ndarray, upper: list[int] | np.ndarray) -> np.ndarray:
        """
        Create a binary mask for an HSV range.

        Args:
            hsv_small (np.ndarray): HSV image at process resolution.
            lower (list[int] | np.ndarray): Lower HSV bound.
            upper (list[int] | np.ndarray): Upper HSV bound.

        Returns:
            np.ndarray: Binary mask.
        """

        return cv2.inRange(hsv_small, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

    def _compute_centroid_small(self, player_mask: np.ndarray) -> tuple[int, int]:
        """
        Compute the centroid of the player mask in process-resolution coordinates.

        Args:
            player_mask (np.ndarray): Player territory mask.

        Returns:
            tuple[int, int]: Centroid in process-resolution pixels.
        """

        moments = cv2.moments(player_mask)
        if moments["m00"] > 0:
            return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        return self.PROCESS_WIDTH // 2, self.PROCESS_HEIGHT // 2

    def _small_to_original(self, point: tuple[int, int], original_shape: tuple[int, int]) -> tuple[int, int]:
        """
        Scale a point from process resolution back to original screenshot coordinates.

        Args:
            point (tuple[int, int]): Point in process resolution.
            original_shape (tuple[int, int]): Original image shape `(height, width)`.

        Returns:
            tuple[int, int]: Point scaled to original resolution.
        """

        original_height, original_width = original_shape
        scale_x = original_width / float(self.PROCESS_WIDTH)
        scale_y = original_height / float(self.PROCESS_HEIGHT)
        return int(point[0] * scale_x), int(point[1] * scale_y)

    def _nearest_target(
        self,
        centroid_small: tuple[int, int],
        target_mask: np.ndarray,
    ) -> tuple[float, tuple[float, float]]:
        """
        Find distance and direction from the player centroid to the nearest target mask pixel.

        Args:
            centroid_small (tuple[int, int]): Player centroid in process resolution.
            target_mask (np.ndarray): Target mask.

        Returns:
            tuple[float, tuple[float, float]]: Distance and normalized direction vector.
        """

        target_points = cv2.findNonZero(target_mask)
        if target_points is None:
            return 999.0, (0.0, 0.0)

        centroid_vector = np.array(centroid_small, dtype=np.float32)
        points = target_points.reshape(-1, 2).astype(np.float32)
        deltas = points - centroid_vector
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        index = int(np.argmin(distances))
        nearest_delta = deltas[index]
        nearest_distance = float(distances[index])
        if nearest_distance <= 0.0:
            return 0.0, (0.0, 0.0)
        return nearest_distance, (
            float(nearest_delta[0] / nearest_distance),
            float(nearest_delta[1] / nearest_distance),
        )

    def _compute_surrounded(self, centroid_small: tuple[int, int], enemy_mask: np.ndarray) -> bool:
        """
        Estimate whether the player is surrounded by sampling 8 compass directions.

        Args:
            centroid_small (tuple[int, int]): Player centroid in process resolution.
            enemy_mask (np.ndarray): Enemy mask at process resolution.

        Returns:
            bool: True when more than 5 of 8 samples land on enemy pixels.
        """

        offsets = [
            (0, -30),
            (21, -21),
            (30, 0),
            (21, 21),
            (0, 30),
            (-21, 21),
            (-30, 0),
            (-21, -21),
        ]
        hits = 0
        for offset_x, offset_y in offsets:
            sample_x = max(0, min(self.PROCESS_WIDTH - 1, centroid_small[0] + offset_x))
            sample_y = max(0, min(self.PROCESS_HEIGHT - 1, centroid_small[1] + offset_y))
            if enemy_mask[sample_y, sample_x] > 0:
                hits += 1
        return hits > 5

    def draw_debug_overlay(self, screenshot_bgr: np.ndarray) -> np.ndarray:
        """
        Draw colored overlays for player, neutral, and enemy masks on a screenshot.

        Args:
            screenshot_bgr (np.ndarray): Original screenshot.

        Returns:
            np.ndarray: Annotated debug image.
        """

        overlay = screenshot_bgr.copy()
        if not self.last_masks:
            return overlay

        original_height, original_width = screenshot_bgr.shape[:2]
        for mask_name, color in (
            ("player", (255, 0, 255)),
            ("neutral", (0, 255, 0)),
            ("enemy", (0, 0, 255)),
        ):
            mask = self.last_masks.get(mask_name)
            if mask is None:
                continue
            resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            color_layer = np.zeros_like(overlay)
            color_layer[resized_mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0.0)

        centroid = self.last_masks.get("centroid")
        if centroid is not None:
            centroid_x, centroid_y = centroid
            cv2.circle(overlay, (int(centroid_x), int(centroid_y)), 6, (255, 255, 255), -1)

        if self.player_color_hsv is not None:
            cv2.putText(
                overlay,
                f"Player HSV: {self.player_color_hsv}",
                (20, original_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def parse(self, screenshot_bgr: np.ndarray | None) -> MapState:
        """
        Parse a raw screenshot into a structured map state.

        Args:
            screenshot_bgr (np.ndarray | None): Raw BGR screenshot.

        Returns:
            MapState: Parsed game state.
        """

        if screenshot_bgr is None or screenshot_bgr.size == 0:
            self.logger.warning("Received empty screenshot during map parsing")
            return self._empty_state()

        if self.player_color_lower is None or self.player_color_upper is None:
            self.detect_and_set_player_color(screenshot_bgr)

        bgr_small, hsv_small, hud_mask_small = self._preprocess(screenshot_bgr)
        valid_mask_small = hud_mask_small.copy()

        if self.player_color_lower is not None and self.player_color_upper is not None:
            player_mask = cv2.inRange(hsv_small, self.player_color_lower, self.player_color_upper)
        else:
            player_mask = np.zeros((self.PROCESS_HEIGHT, self.PROCESS_WIDTH), dtype=np.uint8)

        neutral_profile = COLOR_PROFILES["neutral_land"]
        water_cyan_profile = COLOR_PROFILES["water_cyan"]
        water_dark_profile = COLOR_PROFILES["water_dark"]

        neutral_mask = self._in_range(hsv_small, neutral_profile["lower_hsv"], neutral_profile["upper_hsv"])
        water_cyan_mask = self._in_range(hsv_small, water_cyan_profile["lower_hsv"], water_cyan_profile["upper_hsv"])
        water_dark_mask = self._in_range(hsv_small, water_dark_profile["lower_hsv"], water_dark_profile["upper_hsv"])
        water_mask = cv2.bitwise_or(water_cyan_mask, water_dark_mask)

        player_mask = cv2.bitwise_and(player_mask, valid_mask_small)
        neutral_mask = cv2.bitwise_and(neutral_mask, valid_mask_small)
        water_mask = cv2.bitwise_and(water_mask, valid_mask_small)

        known_mask = cv2.bitwise_or(player_mask, neutral_mask)
        known_mask = cv2.bitwise_or(known_mask, water_mask)
        enemy_mask = cv2.bitwise_not(known_mask)
        enemy_mask = cv2.bitwise_and(enemy_mask, valid_mask_small)

        player_pixels = int(cv2.countNonZero(player_mask))
        neutral_pixels = int(cv2.countNonZero(neutral_mask))
        enemy_pixels = int(cv2.countNonZero(enemy_mask))
        total_land = player_pixels + neutral_pixels + enemy_pixels
        player_ratio = player_pixels / max(total_land, 1)

        centroid_small = self._compute_centroid_small(player_mask)
        centroid_original = self._small_to_original(centroid_small, screenshot_bgr.shape[:2])

        nearest_neutral_distance, nearest_neutral_direction = self._nearest_target(centroid_small, neutral_mask)
        nearest_enemy_distance, nearest_enemy_direction = self._nearest_target(centroid_small, enemy_mask)

        growth = player_pixels - self.prev_player_pixels
        growth_rate = growth / max(self.prev_player_pixels, 1)
        self.prev_player_pixels = player_pixels
        if abs(growth) < 5:
            self.idle_counter += 1
        else:
            self.idle_counter = 0

        is_surrounded = self._compute_surrounded(centroid_small, enemy_mask)
        relative_rank_ratio = player_ratio

        self.last_masks = {
            "player": player_mask.copy(),
            "neutral": neutral_mask.copy(),
            "enemy": enemy_mask.copy(),
            "centroid": np.array(centroid_original, dtype=np.int32),
        }
        self.last_original_shape = screenshot_bgr.shape[:2]

        del bgr_small
        del hsv_small

        return MapState(
            player_territory_pixels=player_pixels,
            enemy_territory_pixels=enemy_pixels,
            neutral_territory_pixels=neutral_pixels,
            player_territory_ratio=player_ratio,
            player_centroid=centroid_original,
            nearest_enemy_distance=nearest_enemy_distance,
            nearest_enemy_direction=nearest_enemy_direction,
            nearest_neutral_distance=nearest_neutral_distance,
            nearest_neutral_direction=nearest_neutral_direction,
            territory_growth_rate=growth_rate,
            is_surrounded=is_surrounded,
            idle_steps=self.idle_counter,
            player_color_hsv=self.player_color_hsv,
            relative_rank_ratio=relative_rank_ratio,
        )
