"""
Reward shaping for Territorial.io Q-learning.
"""

from __future__ import annotations

import logging
from typing import Any

from vision.map_parser import MapState


class RewardCalculator:
    """
    Compute reinforcement-learning rewards from successive map states.
    """

    def __init__(
        self,
        rewards_config: dict[str, Any],
        initial_map_state: MapState | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the reward calculator.

        Args:
            rewards_config (dict[str, Any]): Reward configuration dictionary.
            initial_map_state (MapState | None): Optional initial map state.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.rewards_config = rewards_config
        self.initial_map_state = initial_map_state
        self.logger = logger or logging.getLogger("reward_calculator")
        self.idle_steps = 0
        self.episode_total = 0.0
        self.step_counter = 0

    def leaderboard_rank_reward(self, curr_map_state: MapState) -> float:
        """
        Return a bonus reward based on approximate leaderboard strength.

        Args:
            curr_map_state (MapState): Current parsed map state.

        Returns:
            float: Leaderboard-rank reward bonus.
        """

        ratio = curr_map_state.player_territory_ratio
        if ratio > 0.25:
            return 10.0
        if ratio > 0.10:
            return 5.0
        if ratio > 0.05:
            return 2.0
        return 0.0

    def calculate(
        self,
        prev_map_state: MapState,
        curr_map_state: MapState,
        done: bool,
        won: bool,
    ) -> float:
        """
        Calculate the scalar reward for one state transition.

        Args:
            prev_map_state (MapState): Previous map state.
            curr_map_state (MapState): Current map state.
            done (bool): True if the episode ended this step.
            won (bool): True if the player won the episode.

        Returns:
            float: Reward for the transition.
        """

        self.step_counter += 1
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

        if curr_map_state.player_territory_ratio > 0.02:
            reward += self.rewards_config["top_half_leaderboard_bonus"]

        if self.step_counter % 10 == 0:
            reward += self.leaderboard_rank_reward(curr_map_state)

        if self.idle_steps > 10:
            reward += -0.3
        elif self.idle_steps >= 5:
            reward += self.rewards_config["idle_penalty"]

        if done and won:
            reward += self.rewards_config["win_bonus"]
        elif done and not won:
            reward += self.rewards_config["death_penalty"]

        self.episode_total += reward
        return reward

    def reset(self, initial_map_state: MapState | None = None) -> None:
        """
        Reset reward-tracking state for a new episode.

        Args:
            initial_map_state (MapState | None): Optional initial map state.
        """

        self.initial_map_state = initial_map_state
        self.idle_steps = 0
        self.episode_total = 0.0
        self.step_counter = 0

    def get_episode_total(self) -> float:
        """
        Return the accumulated reward for the current episode.

        Returns:
            float: Episode reward total.
        """

        return self.episode_total
