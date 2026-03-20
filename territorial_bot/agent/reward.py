"""
Reward calculation logic for Territorial.io bot.
"""

from __future__ import annotations

import logging
from typing import Any

from vision.map_parser import MapState


class RewardCalculator:
    """
    Calculate scalar reinforcement-learning rewards from map-state transitions.
    """

    def __init__(
        self,
        rewards_config: dict[str, Any],
        initial_map_state: MapState | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize reward shaping configuration and episode state.

        Args:
            rewards_config (dict[str, Any]): Reward configuration dictionary.
            initial_map_state (MapState | None): Optional initial map state.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.rewards_config = rewards_config
        self.logger = logger or logging.getLogger("reward_calculator")
        self.initial_map_state = initial_map_state
        self.idle_steps = 0
        self.episode_total = 0.0

    def calculate(
        self,
        prev_map_state: MapState,
        curr_map_state: MapState,
        done: bool,
        won: bool,
    ) -> float:
        """
        Calculate reward for a single timestep transition.

        Args:
            prev_map_state (MapState): Map state before action.
            curr_map_state (MapState): Map state after action.
            done (bool): True if episode ended this step.
            won (bool): True if player won the game.

        Returns:
            float: Scalar reward value.
        """

        reward = 0.0
        territory_delta = curr_map_state.player_territory_pixels - prev_map_state.player_territory_pixels
        neutral_delta = prev_map_state.neutral_territory_pixels - curr_map_state.neutral_territory_pixels

        if territory_delta > 0:
            reward += self.rewards_config["territory_gain_reward"] * territory_delta
            self.idle_steps = 0
        elif territory_delta < 0:
            reward += self.rewards_config["territory_loss_penalty"] * abs(territory_delta)
            self.idle_steps = 0
        else:
            self.idle_steps += 1

        reward += self.rewards_config["survival_bonus_per_step"]

        if territory_delta > 0 and curr_map_state.nearest_enemy_distance < prev_map_state.nearest_enemy_distance:
            reward += self.rewards_config["attacking_enemy_bonus"]

        if territory_delta > 0 and neutral_delta > 0:
            reward += self.rewards_config["neutral_capture_reward"]

        if self.idle_steps >= 5:
            reward += self.rewards_config["idle_penalty"]

        if done and won:
            reward += self.rewards_config["win_bonus"]
        elif done and not won:
            reward += self.rewards_config["death_penalty"]

        self.episode_total += reward
        return reward

    def reset(self, initial_map_state: MapState | None = None) -> None:
        """
        Reset internal reward-tracking state for a new episode.

        Args:
            initial_map_state (MapState | None): Optional initial map state.
        """

        self.initial_map_state = initial_map_state
        self.idle_steps = 0
        self.episode_total = 0.0

    def get_episode_total(self) -> float:
        """
        Return the cumulative reward collected during the current episode.

        Returns:
            float: Current episode reward total.
        """

        return self.episode_total
