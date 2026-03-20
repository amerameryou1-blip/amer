"""
State encoding utilities for Territorial.io Q-learning agent.
"""

from __future__ import annotations

import math
from typing import Any

from agent.action_space import ActionSpace
from vision.map_parser import MapState


class StateEncoder:
    """
    Convert parsed map features into a discrete state tuple for tabular Q-learning.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the encoder with configuration and derived dimensions.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
        """

        self.config = config
        self.state_bins = config["agent"]["state_bins"]
        self.max_distance = math.hypot(320, 180)
        self.action_space = ActionSpace(config)

    def _bin_ratio(self, value: float) -> int:
        """
        Bin a ratio value into the configured state bins.

        Args:
            value (float): Ratio value in the range `[0.0, 1.0]`.

        Returns:
            int: Discretized bin index.
        """

        clipped = max(0.0, min(0.999999, value))
        return min(self.state_bins - 1, int(clipped * self.state_bins))

    def _bin_distance(self, value: float) -> int:
        """
        Bin a distance value into the configured state bins.

        Args:
            value (float): Distance measured in analysis-space pixels.

        Returns:
            int: Discretized bin index.
        """

        clipped = max(0.0, min(self.max_distance, value))
        ratio = clipped / self.max_distance
        return min(self.state_bins - 1, int(ratio * self.state_bins))

    def _direction_to_bin(self, direction: tuple[float, float]) -> int:
        """
        Convert a normalized direction vector into one of 8 compass bins.

        Args:
            direction (tuple[float, float]): Direction vector `(dx, dy)`.

        Returns:
            int: Direction bin index in `[0, 7]`.
        """

        dx, dy = direction
        if dx == 0.0 and dy == 0.0:
            return 0

        angle = math.degrees(math.atan2(-dy, dx)) % 360.0
        compass_angle = (90.0 - angle) % 360.0
        return int(((compass_angle + 22.5) % 360.0) // 45.0)

    def _growth_to_bin(self, value: float) -> int:
        """
        Convert continuous territory growth into one of 5 growth categories.

        Args:
            value (float): Relative territory growth value.

        Returns:
            int: Growth-rate category index.
        """

        if value <= -0.10:
            return 0
        if value < -0.01:
            return 1
        if value <= 0.01:
            return 2
        if value < 0.10:
            return 3
        return 4

    def encode(self, map_state: MapState) -> tuple[int, int, int, int, int, int, int]:
        """
        Encode parsed map features into a discrete state tuple.

        Args:
            map_state (MapState): Parsed map state.

        Returns:
            tuple[int, int, int, int, int, int, int]: Hashable discrete state.
        """

        return (
            self._bin_ratio(map_state.player_territory_ratio),
            self._bin_distance(map_state.nearest_enemy_distance),
            self._direction_to_bin(map_state.nearest_enemy_direction),
            self._bin_distance(map_state.nearest_neutral_distance),
            self._direction_to_bin(map_state.nearest_neutral_direction),
            self._growth_to_bin(map_state.territory_growth_rate),
            1 if map_state.is_surrounded else 0,
        )

    def state_space_size(self) -> int:
        """
        Return the total discrete state space size.

        Returns:
            int: Number of possible discrete states.
        """

        return 10 * 10 * 8 * 10 * 8 * 5 * 2

    def action_space_size(self) -> int:
        """
        Return the total number of available actions.

        Returns:
            int: Action count.
        """

        return self.action_space.get_action_count()
