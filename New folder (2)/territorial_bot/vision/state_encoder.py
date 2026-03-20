"""
State encoding for Territorial.io screenshot-grounded Q-learning.
"""

from __future__ import annotations

from typing import Any

from agent.action_space import ActionSpace
from vision.map_parser import MapState


class StateEncoder:
    """
    Convert parsed map state into the 9-element discrete state tuple used by the agent.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the state encoder.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
        """

        self.config = config
        self.action_space = ActionSpace(config)

    def _bin_territory_ratio(self, value: float) -> int:
        """
        Bin the territory ratio into 10 equal-width percentage bins.

        Args:
            value (float): Player land ownership ratio.

        Returns:
            int: Territory-ratio bin.
        """

        clipped = max(0.0, min(0.999999, value))
        return int(clipped * 10.0)

    def _bin_distance(self, value: float) -> int:
        """
        Bin a process-resolution pixel distance into the configured 10 bins.

        Args:
            value (float): Distance in process-resolution pixels.

        Returns:
            int: Distance bin.
        """

        thresholds = [30, 60, 90, 120, 150, 180, 210, 240, 270]
        for index, threshold in enumerate(thresholds):
            if value < threshold:
                return index
        return 9

    def _direction_to_bin(self, direction: tuple[float, float]) -> int:
        """
        Convert a unit vector into one of 8 compass bins.

        Args:
            direction (tuple[float, float]): Direction vector `(dx, dy)`.

        Returns:
            int: Direction bin from 0 to 7.
        """

        dx, dy = direction
        if dy < -0.5 and abs(dx) < 0.5:
            return 0
        if dx > 0.3 and dy < -0.3:
            return 1
        if dx > 0.5 and abs(dy) < 0.5:
            return 2
        if dx > 0.3 and dy > 0.3:
            return 3
        if dy > 0.5 and abs(dx) < 0.5:
            return 4
        if dx < -0.3 and dy > 0.3:
            return 5
        if dx < -0.5 and abs(dy) < 0.5:
            return 6
        if dx < -0.3 and dy < -0.3:
            return 7
        return 0

    def _growth_to_bin(self, growth_rate: float) -> int:
        """
        Convert growth rate into 5 discrete growth categories.

        Args:
            growth_rate (float): Relative territory growth rate.

        Returns:
            int: Growth-rate bin.
        """

        if growth_rate < -0.05:
            return 0
        if growth_rate < -0.005:
            return 1
        if growth_rate < 0.005:
            return 2
        if growth_rate < 0.05:
            return 3
        return 4

    def _idle_steps_to_bin(self, idle_steps: int) -> int:
        """
        Bin idle-step count into 3 categories.

        Args:
            idle_steps (int): Consecutive idle steps.

        Returns:
            int: Idle bin.
        """

        if idle_steps <= 2:
            return 0
        if idle_steps <= 7:
            return 1
        return 2

    def _relative_rank_to_bin(self, map_state: MapState) -> int:
        """
        Approximate leaderboard rank using current territory ratio.

        Args:
            map_state (MapState): Parsed map state.

        Returns:
            int: Relative-rank bin.
        """

        ratio = map_state.player_territory_ratio
        if ratio >= 0.25:
            return 0
        if ratio >= 0.10:
            return 1
        if ratio >= 0.05:
            return 2
        if ratio >= 0.02:
            return 3
        return 4

    def encode(self, map_state: MapState) -> tuple[int, int, int, int, int, int, int, int, int]:
        """
        Encode a parsed map state into the 9-element sparse Q-table key.

        Args:
            map_state (MapState): Parsed map state.

        Returns:
            tuple[int, int, int, int, int, int, int, int, int]: Encoded state tuple.
        """

        return (
            self._bin_territory_ratio(map_state.player_territory_ratio),
            self._bin_distance(map_state.nearest_enemy_distance),
            self._direction_to_bin(map_state.nearest_enemy_direction),
            self._bin_distance(map_state.nearest_neutral_distance),
            self._direction_to_bin(map_state.nearest_neutral_direction),
            self._growth_to_bin(map_state.territory_growth_rate),
            1 if map_state.is_surrounded else 0,
            self._idle_steps_to_bin(map_state.idle_steps),
            self._relative_rank_to_bin(map_state),
        )

    def state_space_size(self) -> int:
        """
        Return the total sparse state-space size.

        Returns:
            int: Total number of possible states.
        """

        return 10 * 10 * 8 * 10 * 8 * 5 * 2 * 3 * 5

    def action_space_size(self) -> int:
        """
        Return the total number of available actions.

        Returns:
            int: Action count.
        """

        return self.action_space.get_action_count()
