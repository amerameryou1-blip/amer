"""
Action-space definition for the rebuilt Territorial.io bot.
"""

from __future__ import annotations

from typing import Any


class ActionSpace:
    """
    Describe the 73-action Territorial.io policy space.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the action-space metadata.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
        """

        self.config = config
        self.actions_config = config["actions"]
        self.grid_size = int(self.actions_config["grid_size"])
        self.grid_action_count = self.grid_size * self.grid_size

    def get_action_coordinates(
        self,
        action_id: int,
        player_centroid: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """
        Resolve an action ID into a representative screen coordinate.

        Args:
            action_id (int): Discrete action index.
            player_centroid (tuple[int, int] | None): Optional player centroid in screen coordinates.

        Returns:
            tuple[int, int]: Screen coordinate representative for the action.
        """

        if 0 <= action_id < self.grid_action_count:
            cell_x = action_id % self.grid_size
            cell_y = action_id // self.grid_size
            pixel_x = int(self.actions_config["playable_x_start"]) + cell_x * 133 + 66
            pixel_y = int(self.actions_config["playable_y_start"]) + cell_y * 80 + 40
            return pixel_x, pixel_y

        if 64 <= action_id <= 71:
            centroid_x, centroid_y = player_centroid or (
                self.config["game"]["window_width"] // 2,
                self.config["game"]["window_height"] // 2,
            )
            offset = int(self.actions_config["directional_offset_px"])
            offsets = [
                (0, -offset),
                (offset, -offset),
                (offset, 0),
                (offset, offset),
                (0, offset),
                (-offset, offset),
                (-offset, 0),
                (-offset, -offset),
            ]
            delta_x, delta_y = offsets[action_id - 64]
            return centroid_x + delta_x, centroid_y + delta_y

        if action_id == 72:
            return -1, -1

        raise ValueError(f"Unsupported action id: {action_id}")

    def get_action_count(self) -> int:
        """
        Return the total number of actions.

        Returns:
            int: Total action count.
        """

        return self.grid_action_count + 8 + 1

    def describe_action(self, action_id: int) -> str:
        """
        Return a human-readable description for an action.

        Args:
            action_id (int): Discrete action index.

        Returns:
            str: Action description.
        """

        if 0 <= action_id < self.grid_action_count:
            cell_x = action_id % self.grid_size
            cell_y = action_id // self.grid_size
            pixel_x, pixel_y = self.get_action_coordinates(action_id)
            return f"grid click ({cell_x},{cell_y}) -> ({pixel_x}, {pixel_y})"
        if 64 <= action_id <= 71:
            directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            return f"directional move+attack {directions[action_id - 64]}"
        if action_id == 72:
            return "spacebar attack"
        raise ValueError(f"Unsupported action id: {action_id}")
