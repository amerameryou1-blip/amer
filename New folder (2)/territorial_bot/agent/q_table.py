"""
Sparse Q-table storage and persistence for Territorial.io bot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from utils.file_utils import safe_pickle_load, safe_pickle_save


class QTable:
    """
    Sparse mapping from discrete states to arrays of action values.
    """

    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize an empty sparse Q-table.

        Args:
            state_space_size (int): Total number of possible states.
            action_space_size (int): Total number of available actions.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.logger = logger or logging.getLogger("q_table")
        self.table: dict[tuple[int, ...], np.ndarray] = {}

    def get(self, state: tuple[int, ...]) -> np.ndarray:
        """
        Return the Q-value array for a state, creating it lazily if needed.

        Args:
            state (tuple[int, ...]): Encoded discrete state.

        Returns:
            np.ndarray: Q-values for all actions.
        """

        if state not in self.table:
            self.table[state] = np.zeros(self.action_space_size, dtype=np.float32)
        return self.table[state]

    def update(self, state: tuple[int, ...], action: int, value: float) -> None:
        """
        Update a single state-action value.

        Args:
            state (tuple[int, ...]): Encoded discrete state.
            action (int): Action index.
            value (float): New Q-value.
        """

        values = self.get(state)
        values[action] = float(value)

    def best_action(self, state: tuple[int, ...]) -> int:
        """
        Return the greedy action for a given state.

        Args:
            state (tuple[int, ...]): Encoded discrete state.

        Returns:
            int: Action index with highest Q-value.
        """

        return int(np.argmax(self.get(state)))

    def resize(self, new_action_space_size: int) -> None:
        """
        Resize all stored action-value arrays to a new action-space size.

        Args:
            new_action_space_size (int): Updated action count.
        """

        if new_action_space_size == self.action_space_size:
            return

        for state, values in list(self.table.items()):
            resized = np.zeros(new_action_space_size, dtype=np.float32)
            common_size = min(len(values), new_action_space_size)
            resized[:common_size] = values[:common_size]
            self.table[state] = resized
        self.action_space_size = new_action_space_size

    def save(self, path: str | Path) -> None:
        """
        Save the Q-table to disk using atomic replacement.

        Args:
            path (str | Path): Destination file path.
        """

        payload = {
            "state_space_size": self.state_space_size,
            "action_space_size": self.action_space_size,
            "table": self.table,
        }
        safe_pickle_save(payload, path)

    def load(self, path: str | Path) -> bool:
        """
        Load the Q-table from disk if a valid checkpoint exists.

        Args:
            path (str | Path): Source checkpoint path.

        Returns:
            bool: True when a checkpoint was loaded successfully.
        """

        payload = safe_pickle_load(path)
        if payload is None:
            return False

        try:
            target_action_space_size = self.action_space_size
            table = payload.get("table", {})
            action_space_size = int(payload.get("action_space_size", self.action_space_size))
            state_space_size = int(payload.get("state_space_size", self.state_space_size))
            if not isinstance(table, dict):
                raise ValueError("Checkpoint table payload is not a dictionary")

            self.table = {
                tuple(state): np.array(values, dtype=np.float32)
                for state, values in table.items()
            }
            self.state_space_size = state_space_size
            self.action_space_size = action_space_size
            self.resize(target_action_space_size)
            self.logger.info("Loaded Q-table checkpoint from %s with %s states", path, len(self.table))
            return True
        except Exception as exc:
            self.table = {}
            self.logger.warning("Q-table checkpoint is corrupted or incompatible at %s: %s", path, exc)
            return False

    def get_stats(self) -> dict[str, float]:
        """
        Compute summary statistics for the stored Q-values.

        Returns:
            dict[str, float]: Visited-state count, mean Q-value, and max Q-value.
        """

        if not self.table:
            return {"total_states_visited": 0, "mean_q_value": 0.0, "max_q_value": 0.0}

        total_sum = 0.0
        total_count = 0
        max_value = float("-inf")
        for values in self.table.values():
            total_sum += float(np.sum(values))
            total_count += int(values.size)
            max_value = max(max_value, float(np.max(values)))

        mean_value = total_sum / max(total_count, 1)
        return {
            "total_states_visited": len(self.table),
            "mean_q_value": mean_value,
            "max_q_value": max_value,
        }

    def merge(self, other_q_table: "QTable") -> None:
        """
        Merge another Q-table into this one by averaging overlapping states.

        Args:
            other_q_table (QTable): Another Q-table to merge.
        """

        self.resize(max(self.action_space_size, other_q_table.action_space_size))
        for state, other_values in other_q_table.table.items():
            if state not in self.table:
                self.table[state] = other_values.copy()
                continue
            current_values = self.table[state]
            common_size = min(len(current_values), len(other_values))
            current_values[:common_size] = (current_values[:common_size] + other_values[:common_size]) / 2.0
