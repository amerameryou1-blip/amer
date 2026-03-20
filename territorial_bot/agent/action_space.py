"""
Action-space definition for Territorial.io bot.
"""

from __future__ import annotations

from typing import Any


class ActionSpace:
    """
    Map discrete action IDs to screen coordinates on the Territorial.io map.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the click grid and directional nudge configuration.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
        """

        self.config = config
        self.actions_config = config["actions"]
        self.map_region = config["vision"]["map_region"]
        self.grid_size = self.actions_config["click_grid_size"]
        self.move_directions = self.actions_config["move_directions"]
        self.grid_action_count = self.grid_size * self.grid_size
        self.analysis_width = 320
        self.analysis_height = 180
        self.direction_vectors = {
            "up": (0.0, -1.0),
            "down": (0.0, 1.0),
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
            "up_left": (-0.7071, -0.7071),
            "up_right": (0.7071, -0.7071),
            "down_left": (-0.7071, 0.7071),
            "down_right": (0.7071, 0.7071),
        }

    def _map_dimensions(self) -> tuple[int, int, int, int]:
        """
        Return the configured map region dimensions.

        Returns:
            tuple[int, int, int, int]: `(x, y, width, height)` map region.
        """

        x, y, width, height = self.map_region
        return int(x), int(y), int(width), int(height)

    def _analysis_to_screen(self, point: tuple[int, int] | None) -> tuple[int, int]:
        """
        Convert analysis-space centroid coordinates into screen coordinates.

        Args:
            point (tuple[int, int] | None): Point in 320x180 analysis space.

        Returns:
            tuple[int, int]: Absolute screen coordinate on the map region.
        """

        x, y, width, height = self._map_dimensions()
        if point is None:
            return x + width // 2, y + height // 2
        src_x, src_y = point
        norm_x = max(0.0, min(1.0, src_x / max(self.analysis_width - 1, 1)))
        norm_y = max(0.0, min(1.0, src_y / max(self.analysis_height - 1, 1)))
        screen_x = x + int(norm_x * width)
        screen_y = y + int(norm_y * height)
        return screen_x, screen_y

    def get_action_coordinates(
        self,
        action_id: int,
        player_centroid: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """
        Resolve an action ID into an absolute screen coordinate.

        Args:
            action_id (int): Discrete action index.
            player_centroid (tuple[int, int] | None): Latest player centroid in analysis space.

        Returns:
            tuple[int, int]: Absolute screen coordinate.
        """

        x, y, width, height = self._map_dimensions()
        if action_id < self.grid_action_count:
            row = action_id // self.grid_size
            column = action_id % self.grid_size
            cell_width = width / self.grid_size
            cell_height = height / self.grid_size
            target_x = int(x + (column + 0.5) * cell_width)
            target_y = int(y + (row + 0.5) * cell_height)
            return target_x, target_y

        direction_index = action_id - self.grid_action_count
        if direction_index >= len(self.move_directions):
            raise ValueError(f"Action ID {action_id} exceeds action space size {self.get_action_count()}")

        direction_name = self.move_directions[direction_index]
        origin_x, origin_y = self._analysis_to_screen(player_centroid)
        vector_x, vector_y = self.direction_vectors[direction_name]
        nudge_distance = int(min(width, height) * 0.12)
        target_x = int(origin_x + vector_x * nudge_distance)
        target_y = int(origin_y + vector_y * nudge_distance)
        target_x = max(x, min(x + width - 1, target_x))
        target_y = max(y, min(y + height - 1, target_y))
        return target_x, target_y

    def get_action_count(self) -> int:
        """
        Return the total number of discrete actions.

        Returns:
            int: Grid clicks plus directional nudges.
        """

        return self.grid_action_count + len(self.move_directions)

    def describe_action(self, action_id: int) -> str:
        """
        Return a human-readable description for an action ID.

        Args:
            action_id (int): Discrete action index.

        Returns:
            str: Description of the action target.
        """

        if action_id < self.grid_action_count:
            row = action_id // self.grid_size
            column = action_id % self.grid_size
            x, y = self.get_action_coordinates(action_id)
            return f"click at grid ({row},{column}) = pixel ({x}, {y})"

        direction_index = action_id - self.grid_action_count
        direction_name = self.move_directions[direction_index]
        x, y = self.get_action_coordinates(action_id)
        return f"directional nudge '{direction_name}' = pixel ({x}, {y})"
