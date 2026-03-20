"""
Color profiles and HUD masking helpers for Territorial.io vision.
"""

from __future__ import annotations

import cv2
import numpy as np


COLOR_PROFILES = {
    "water_cyan": {
        "lower_hsv": [85, 150, 150],
        "upper_hsv": [105, 255, 255],
        "description": "Bright turquoise/cyan water",
    },
    "water_dark": {
        "lower_hsv": [100, 80, 50],
        "upper_hsv": [130, 255, 180],
        "description": "Dark blue/navy water",
    },
    "neutral_land": {
        "lower_hsv": [35, 80, 120],
        "upper_hsv": [75, 220, 255],
        "description": "Bright lime green neutral territory",
    },
    "hud_leaderboard": {
        "region": [0, 0, 310, 340],
        "description": "Leaderboard panel top-left",
    },
    "hud_stats": {
        "region": [1200, 0, 166, 160],
        "description": "Stats panel top-right",
    },
    "hud_troops_counter": {
        "region": [580, 20, 370, 70],
        "description": "Troops count top-center",
    },
    "hud_troop_bar": {
        "region": [450, 730, 630, 38],
        "description": "Troop deployment bar bottom",
        "player_color_sample_pixel": [760, 748],
    },
    "hud_zoom": {
        "region": [1300, 330, 66, 160],
        "description": "Zoom buttons on the right edge",
    },
    "hud_exit": {
        "region": [0, 720, 60, 48],
        "description": "Exit game button bottom-left",
    },
}


def _scaled_region(region: list[int], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """
    Scale a 1366x768 reference region into the current image resolution.

    Args:
        region (list[int]): `[x, y, width, height]` region at 1366x768 reference size.
        image_shape (tuple[int, ...]): Current image shape.

    Returns:
        tuple[int, int, int, int]: Scaled region clipped to the image size.
    """

    height, width = image_shape[:2]
    scale_x = width / 1366.0
    scale_y = height / 768.0
    x, y, region_width, region_height = region
    scaled_x = int(round(x * scale_x))
    scaled_y = int(round(y * scale_y))
    scaled_width = int(round(region_width * scale_x))
    scaled_height = int(round(region_height * scale_y))
    return scaled_x, scaled_y, scaled_width, scaled_height


def get_hud_mask(image_shape: tuple[int, ...]) -> np.ndarray:
    """
    Return a binary mask where HUD pixels are black and playable pixels are white.

    Args:
        image_shape (tuple[int, ...]): Image shape `(height, width, channels)`.

    Returns:
        np.ndarray: HUD mask with HUD regions set to 0 and map regions set to 255.
    """

    height, width = image_shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    hud_regions = [
        COLOR_PROFILES["hud_leaderboard"]["region"],
        COLOR_PROFILES["hud_stats"]["region"],
        COLOR_PROFILES["hud_troops_counter"]["region"],
        COLOR_PROFILES["hud_troop_bar"]["region"],
        COLOR_PROFILES["hud_zoom"]["region"],
        COLOR_PROFILES["hud_exit"]["region"],
    ]
    for region in hud_regions:
        x, y, region_width, region_height = _scaled_region(region, image_shape)
        x2 = min(x + region_width, width)
        y2 = min(y + region_height, height)
        mask[max(0, y):y2, max(0, x):x2] = 0
    return mask


def detect_player_color_from_troopbar(
    screenshot_bgr: np.ndarray,
    sample_pixel: tuple[int, int] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]] | None:
    """
    Detect the player's current territory color by sampling the troop bar fill.

    Args:
        screenshot_bgr (np.ndarray): Raw BGR screenshot.
        sample_pixel (tuple[int, int] | list[int] | None): Optional sample coordinate override.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[int, int, int]] | None:
            Lower HSV bound, upper HSV bound, and median HSV color, or None on failure.
    """

    height, width = screenshot_bgr.shape[:2]
    sample_x, sample_y = tuple(sample_pixel or COLOR_PROFILES["hud_troop_bar"]["player_color_sample_pixel"])
    if sample_x >= width or sample_y >= height:
        sample_x = int(width * 0.50)
        sample_y = int(height * 0.975)

    region = screenshot_bgr[
        max(0, sample_y - 2):min(height, sample_y + 3),
        max(0, sample_x - 5):min(width, sample_x + 5),
    ]
    if region.size == 0:
        return None

    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    median_hsv = np.median(hsv_region.reshape(-1, 3), axis=0).astype(np.uint8)
    h_val, s_val, v_val = [int(channel) for channel in median_hsv]
    if s_val < 40 or v_val < 40:
        return None

    h_tol, s_tol, v_tol = 12, 60, 60
    lower = np.array(
        [
            max(0, h_val - h_tol),
            max(0, s_val - s_tol),
            max(0, v_val - v_tol),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            min(179, h_val + h_tol),
            min(255, s_val + s_tol),
            min(255, v_val + v_tol),
        ],
        dtype=np.uint8,
    )
    return lower, upper, (h_val, s_val, v_val)


def is_water_pixel(hsv_pixel: tuple[int, int, int] | np.ndarray) -> bool:
    """
    Return True when a single HSV pixel matches a water color.

    Args:
        hsv_pixel (tuple[int, int, int] | np.ndarray): HSV pixel.

    Returns:
        bool: True if the pixel is uncapturable water.
    """

    h, s, v = [int(channel) for channel in hsv_pixel]
    cyan = (85 <= h <= 105) and (s >= 150) and (v >= 150)
    dark_blue = (100 <= h <= 130) and (s >= 80) and (v <= 180)
    return cyan or dark_blue


def is_neutral_pixel(hsv_pixel: tuple[int, int, int] | np.ndarray) -> bool:
    """
    Return True when a single HSV pixel matches neutral capturable land.

    Args:
        hsv_pixel (tuple[int, int, int] | np.ndarray): HSV pixel.

    Returns:
        bool: True if the pixel is neutral territory.
    """

    h, s, v = [int(channel) for channel in hsv_pixel]
    return (35 <= h <= 75) and (80 <= s <= 220) and (120 <= v <= 255)
