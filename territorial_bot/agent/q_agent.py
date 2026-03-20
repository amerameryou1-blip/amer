"""
Q-learning agent logic for Territorial.io bot.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from agent.action_space import ActionSpace
from agent.q_table import QTable


class QAgent:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.
    """

    def __init__(
        self,
        config: dict[str, Any],
        q_table: QTable,
        action_space: ActionSpace,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the agent with Q-learning hyperparameters.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            q_table (QTable): Backing sparse Q-table.
            action_space (ActionSpace): Discrete action-space mapper.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.config = config
        self.agent_config = config["agent"]
        self.q_table = q_table
        self.action_space = action_space
        self.logger = logger or logging.getLogger("q_agent")
        self.epsilon = float(self.agent_config["epsilon_start"])
        self.episode_count = 0
        self.step_count = 0
        self.recent_rewards: list[float] = []
        self.recent_territory_ratios: list[float] = []

    def select_action(self, state: tuple[int, ...]) -> int:
        """
        Choose an action using epsilon-greedy exploration.

        Args:
            state (tuple[int, ...]): Encoded environment state.

        Returns:
            int: Selected action index.
        """

        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(self.action_space.get_action_count())
        return self.q_table.best_action(state)

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        """
        Apply the tabular Q-learning update rule.

        Args:
            state (tuple[int, ...]): Previous state.
            action (int): Action taken.
            reward (float): Immediate reward.
            next_state (tuple[int, ...]): Resulting next state.
            done (bool): True when the episode terminated.
        """

        alpha = self.agent_config["learning_rate"]
        gamma = self.agent_config["discount_factor"]
        current_values = self.q_table.get(state)
        current_q = float(current_values[action])
        next_max = 0.0 if done else float(np.max(self.q_table.get(next_state)))
        updated_q = current_q + alpha * (reward + gamma * next_max - current_q)
        self.q_table.update(state, action, updated_q)

    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate according to the configured strategy.
        """

        epsilon_min = self.agent_config["epsilon_min"]
        decay = self.agent_config["epsilon_decay"]
        strategy = self.agent_config["epsilon_decay_strategy"]

        if strategy == "exponential":
            self.epsilon = max(epsilon_min, self.epsilon * decay)
            return

        linear_step = (1.0 - decay) / max(self.config["game"]["max_episodes"], 1)
        self.epsilon = max(epsilon_min, self.epsilon - linear_step)

    def complete_episode(self, total_reward: float, territory_ratio: float, decay: bool = True) -> None:
        """
        Record episode statistics and optionally decay epsilon.

        Args:
            total_reward (float): Episode reward total.
            territory_ratio (float): Final territory ratio.
            decay (bool): Whether to decay epsilon after the episode.
        """

        self.episode_count += 1
        self.recent_rewards.append(total_reward)
        self.recent_territory_ratios.append(territory_ratio)
        self.recent_rewards = self.recent_rewards[-10:]
        self.recent_territory_ratios = self.recent_territory_ratios[-10:]

        if decay:
            self.decay_epsilon()

        if self.episode_count % 10 == 0:
            avg_reward = sum(self.recent_rewards) / max(len(self.recent_rewards), 1)
            avg_ratio = sum(self.recent_territory_ratios) / max(len(self.recent_territory_ratios), 1)
            self.logger.info(
                "Agent summary after %s episodes | avg_reward=%.3f | epsilon=%.4f | avg_territory_ratio=%.4f",
                self.episode_count,
                avg_reward,
                self.epsilon,
                avg_ratio,
            )

    def get_stats(self) -> dict[str, Any]:
        """
        Return agent statistics and Q-table summary information.

        Returns:
            dict[str, Any]: Agent and Q-table statistics.
        """

        return {
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "q_table_stats": self.q_table.get_stats(),
        }
