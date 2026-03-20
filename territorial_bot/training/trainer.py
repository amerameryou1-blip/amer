"""
Single-worker training loop for Territorial.io bot.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError

from vision.screenshot import ScreenshotCapture
from utils.timer import GameTimer


class Trainer:
    """
    Run Territorial.io episodes and update the Q-learning agent.
    """

    def __init__(
        self,
        config: dict[str, Any],
        agent: Any,
        browser_manager: Any,
        game_launcher: Any,
        game_controller: Any,
        map_parser: Any,
        state_encoder: Any,
        reward_calculator: Any,
        episode_logger: Any,
        worker_id: int = 0,
        training_enabled: bool = True,
        screenshot_capture: ScreenshotCapture | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize a training loop instance with all runtime dependencies.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            agent (Any): Q-learning agent.
            browser_manager (Any): Browser manager instance.
            game_launcher (Any): Game launcher instance.
            game_controller (Any): Controller for mouse and keyboard actions.
            map_parser (Any): Computer-vision map parser.
            state_encoder (Any): State encoder instance.
            reward_calculator (Any): Reward calculator instance.
            episode_logger (Any): Episode logger instance.
            worker_id (int): Worker identifier for logs and checkpoints.
            training_enabled (bool): Whether Q-values should be updated.
            screenshot_capture (ScreenshotCapture | None): Optional screenshot capture helper.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.config = config
        self.agent = agent
        self.browser_manager = browser_manager
        self.game_launcher = game_launcher
        self.game_controller = game_controller
        self.map_parser = map_parser
        self.state_encoder = state_encoder
        self.reward_calculator = reward_calculator
        self.episode_logger = episode_logger
        self.worker_id = worker_id
        self.training_enabled = training_enabled
        self.logger = logger or logging.getLogger("trainer")
        self.screenshot_capture = screenshot_capture or ScreenshotCapture(config, logger=self.logger)
        self.game_config = config["game"]
        self.agent_config = config["agent"]
        self.restart_on_death = self.game_config["restart_on_death"]

    def _capture_valid_frame(self, retries: int = 3) -> Any | None:
        """
        Capture a screenshot with limited retries to tolerate transient failures.

        Args:
            retries (int): Maximum number of attempts.

        Returns:
            Any | None: Captured BGR frame, or None if all attempts failed.
        """

        page = self.browser_manager.get_page()
        for attempt in range(1, retries + 1):
            frame = self.screenshot_capture.capture(page)
            if frame is not None and frame.size > 0:
                return frame
            self.logger.warning("Screenshot capture returned blank frame on attempt %s/%s", attempt, retries)
            time.sleep(0.2)
        return None

    def _refresh_page_binding(self) -> None:
        """
        Refresh page-bound helpers after a browser restart.
        """

        self.game_controller.set_page(self.browser_manager.get_page())

    def save_q_table(self, path: str | Path | None = None) -> Path:
        """
        Save the Q-table checkpoint with a backup fallback path on failure.

        Args:
            path (str | Path | None): Optional override checkpoint path.

        Returns:
            Path: Path of the primary or backup checkpoint that was saved.
        """

        target_path = Path(path or self.agent_config["q_table_path"])
        try:
            self.agent.q_table.save(target_path)
            self.logger.info("Saved Q-table checkpoint to %s", target_path)
            return target_path
        except Exception as exc:
            self.logger.exception("Primary Q-table save failed at %s: %s", target_path, exc)
            backup_path = target_path.with_name(f"{target_path.stem}_backup_worker{self.worker_id}{target_path.suffix}")
            self.agent.q_table.save(backup_path)
            self.logger.warning("Saved backup Q-table checkpoint to %s", backup_path)
            return backup_path

    def run_episode(self) -> dict[str, Any]:
        """
        Run a single training or evaluation episode.

        Returns:
            dict[str, Any]: Episode statistics summary.
        """

        self._refresh_page_binding()
        self.game_launcher.handle_menu_state()
        if not self.game_launcher.start_match():
            raise RuntimeError("Failed to start a new Territorial.io match")

        time.sleep(self.game_config["game_start_wait_s"])
        episode_timer = GameTimer()
        self.reward_calculator.reset()
        self.map_parser.reset()

        initial_frame = self._capture_valid_frame(retries=5)
        if initial_frame is None:
            raise RuntimeError("Unable to capture initial frame for new episode")

        current_map_state = self.map_parser.parse(initial_frame)
        self.reward_calculator.reset(current_map_state)
        current_state = self.state_encoder.encode(current_map_state)

        total_steps = 0
        done = False
        won = False

        for step in range(1, self.game_config["max_episode_steps"] + 1):
            total_steps = step
            self.game_controller.set_player_centroid(current_map_state.player_centroid)
            action = self.agent.select_action(current_state)
            action_x, action_y = self.game_controller.execute_action(action)

            time.sleep(self.game_config["screenshot_interval_ms"] / 1000.0)
            next_frame = self._capture_valid_frame()
            if next_frame is None:
                self.logger.warning("Skipping step %s because screenshot capture failed", step)
                continue

            try:
                done = self.game_launcher.detect_game_over()
            except Exception:
                done = False
            won = self.game_launcher.detect_victory() if done else False

            next_map_state = self.map_parser.parse(next_frame)
            next_state = self.state_encoder.encode(next_map_state)
            reward = self.reward_calculator.calculate(current_map_state, next_map_state, done, won)

            if self.training_enabled:
                self.agent.update(current_state, action, reward, next_state, done)

            self.episode_logger.log_step(
                {
                    "episode": self.agent.episode_count + 1,
                    "worker_id": self.worker_id,
                    "step": step,
                    "action_id": action,
                    "action_x": action_x,
                    "action_y": action_y,
                    "reward": reward,
                    "done": done,
                    "won": won,
                    "territory_ratio": next_map_state.player_territory_ratio,
                }
            )

            current_map_state = next_map_state
            current_state = next_state

            if done:
                break

        total_reward = self.reward_calculator.get_episode_total()
        final_ratio = current_map_state.player_territory_ratio
        self.agent.complete_episode(total_reward, final_ratio, decay=self.training_enabled)

        episode_number = self.agent.episode_count
        if self.training_enabled and episode_number % self.agent_config["q_table_save_interval"] == 0:
            self.save_q_table()

        stats = {
            "episode": episode_number,
            "worker_id": self.worker_id,
            "total_steps": total_steps,
            "total_reward": total_reward,
            "final_territory_ratio": final_ratio,
            "epsilon": self.agent.epsilon,
            "duration_seconds": episode_timer.elapsed_seconds(),
            "won": won,
        }
        self.episode_logger.log_episode(stats)

        if self.restart_on_death and (done or total_steps >= self.game_config["max_episode_steps"]):
            self.browser_manager.restart()
            self._refresh_page_binding()

        return stats

    def run(self, num_episodes: int) -> list[dict[str, Any]]:
        """
        Run multiple episodes with crash recovery and checkpoint persistence.

        Args:
            num_episodes (int): Number of episodes to execute.

        Returns:
            list[dict[str, Any]]: Collected episode summaries.
        """

        episode_stats: list[dict[str, Any]] = []
        for _ in range(num_episodes):
            try:
                stats = self.run_episode()
                episode_stats.append(stats)
            except KeyboardInterrupt:
                self.logger.warning("KeyboardInterrupt received; saving Q-table before exit")
                self.save_q_table()
                raise
            except PlaywrightError as exc:
                self.logger.exception("Playwright error during episode: %s", exc)
                self.browser_manager.restart()
                self._refresh_page_binding()
            except Exception as exc:
                self.logger.exception("Unhandled training error: %s", exc)
                self.browser_manager.restart()
                self._refresh_page_binding()
        return episode_stats
