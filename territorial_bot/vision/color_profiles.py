"""
HSV color profiles used to segment the Territorial.io map.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


COLOR_PROFILES: dict[str, dict[str, list[int]]] = {
    "player_territory": {
        "lower": [92, 70, 60],
        "upper": [132, 255, 255],
    },
    "enemy_territory": {
        "lower": [0, 110, 70],
        "upper": [25, 255, 255],
    },
    "neutral_territory": {
        "lower": [5, 10, 70],
        "upper": [40, 90, 220],
    },
    "background": {
        "lower": [0, 0, 0],
        "upper": [180, 80, 70],
    },
    "ui_elements": {
        "lower": [0, 0, 180],
        "upper": [180, 70, 255],
    },
}


def detect_color_mask(
    img: np.ndarray,
    profile_name: str,
    tolerance: int = 0,
    profiles: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Detect a binary mask for a configured HSV profile.

    Args:
        img (np.ndarray): Input BGR image.
        profile_name (str): Named color profile to apply.
        tolerance (int): Additional tolerance expanded onto profile bounds.
        profiles (dict[str, Any] | None): Optional profile override dictionary.

    Returns:
        np.ndarray: Binary mask image.
    """

    active_profiles = profiles or COLOR_PROFILES
    profile = active_profiles[profile_name]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(profile["lower"], dtype=np.int16) - tolerance
    upper = np.array(profile["upper"], dtype=np.int16) + tolerance
    lower = np.clip(lower, [0, 0, 0], [180, 255, 255]).astype(np.uint8)
    upper = np.clip(upper, [0, 0, 0], [180, 255, 255]).astype(np.uint8)
    return cv2.inRange(hsv, lower, upper)
