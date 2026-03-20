"""
Single-worker Territorial.io training loop.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError

from utils.timer import GameTimer
from vision.screenshot import ScreenshotCapture


class Trainer:
    """
    Run Territorial.io episodes using screenshot-grounded state updates.
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
        Initialize the trainer.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            agent (Any): Q-learning agent.
            browser_manager (Any): Browser manager instance.
            game_launcher (Any): Game launcher instance.
            game_controller (Any): Mouse and keyboard controller.
            map_parser (Any): Screenshot map parser.
            state_encoder (Any): Discrete state encoder.
            reward_calculator (Any): Reward calculator.
            episode_logger (Any): Episode logger.
            worker_id (int): Worker identifier.
            training_enabled (bool): Whether to update the Q-table.
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
        self.debug_config = config.get("debug", {})

    def _refresh_page_binding(self) -> None:
        """
        Refresh helpers that hold a direct page reference.
        """

        page = self.browser_manager.get_page()
        self.game_controller.set_page(page)
        if hasattr(self.game_launcher, "set_page"):
            self.game_launcher.set_page(page)

    def _capture_frame(self) -> Any | None:
        """
        Capture a screenshot from the active page.

        Returns:
            Any | None: Screenshot array or None on failure.
        """

        return self.screenshot_capture.capture(self.browser_manager.get_page())

    def _save_debug_overlay(self, screenshot_bgr: Any, episode_number: int, step: int) -> None:
        """
        Save a debug overlay frame when debug vision is enabled.

        Args:
            screenshot_bgr (Any): Raw BGR screenshot.
            episode_number (int): Episode number for naming.
            step (int): Step number for naming.
        """

        overlay = self.map_parser.draw_debug_overlay(screenshot_bgr)
        saved_path = self.screenshot_capture.save_debug_frame(overlay, episode_number, step)
        self.logger.info("Saved debug overlay frame to %s", saved_path)

    def _exit_to_menu(self) -> None:
        """
        Click the exit button to return to the main menu.
        """

        self.browser_manager.get_page().mouse.click(30, 748)
        time.sleep(1.0)

    def save_q_table(self, path: str | Path | None = None) -> Path:
        """
        Save the Q-table checkpoint, falling back to a backup path on failure.

        Args:
            path (str | Path | None): Optional save path override.

        Returns:
            Path: Final checkpoint path that was written.
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
        Run one full Territorial.io episode from menu entry to cleanup.

        Returns:
            dict[str, Any]: Episode statistics, or an empty dict if startup failed.
        """

        self._refresh_page_binding()
        self.map_parser.reset()
        self.reward_calculator.reset()
        episode_step = 0
        total_reward = 0.0
        done = False
        timer = GameTimer()
        episode_number = self.agent.episode_count + 1

        if not self.game_launcher.navigate_to_game():
            self.logger.warning("Episode %s aborted: failed to navigate to game", episode_number)
            return {}
        self.game_launcher.set_player_name(self.game_config["player_name"])
        self.game_launcher.click_multiplayer()
        if not self.game_launcher.wait_for_game_start():
            self.logger.warning("Episode %s aborted: game start not detected", episode_number)
            return {}

        player_color_detected = False
        for attempt in range(1, 6):
            color_frame = self._capture_frame()
            if color_frame is None:
                time.sleep(1.0)
                continue
            if self.map_parser.detect_and_set_player_color(color_frame):
                player_color_detected = True
                break
            self.logger.warning("Player color detection retry %s/5 failed", attempt)
            time.sleep(1.0)

        if not player_color_detected:
            self.logger.error("Episode %s aborted: player color detection failed after 5 attempts", episode_number)
            return {}

        time.sleep(2.0)
        screenshot = self._capture_frame()
        if screenshot is None:
            self.logger.warning("Episode %s aborted: missing initial screenshot", episode_number)
            return {}

        current_map_state = self.map_parser.parse(screenshot)
        current_state = self.state_encoder.encode(current_map_state)
        self.reward_calculator.reset(current_map_state)
        self.logger.info(
            "Startup summary | resume_episode=%s | player_color_hsv=%s | territory_pixels=%s",
            episode_number,
            current_map_state.player_color_hsv,
            current_map_state.player_territory_pixels,
        )

        for step in range(self.game_config["max_episode_steps"]):
            action = self.agent.select_action(current_state)
            action_x, action_y = self.game_controller.execute_action(action, current_map_state)
            time.sleep(self.game_config["screenshot_interval_ms"] / 1000.0)

            screenshot = self._capture_frame()
            if screenshot is None:
                continue

            done = self.game_launcher.detect_defeat(screenshot)
            new_map_state = self.map_parser.parse(screenshot)
            new_state = self.state_encoder.encode(new_map_state)
            reward = self.reward_calculator.calculate(current_map_state, new_map_state, done, won=False)

            if self.training_enabled:
                self.agent.update(current_state, action, reward, new_state, done)

            current_state = new_state
            current_map_state = new_map_state
            total_reward += reward
            episode_step = step + 1

            self.episode_logger.log_step(
                {
                    "episode": episode_number,
                    "worker_id": self.worker_id,
                    "step": episode_step,
                    "action_id": action,
                    "action_x": action_x,
                    "action_y": action_y,
                    "reward": reward,
                    "territory_ratio": current_map_state.player_territory_ratio,
                    "player_pixels": current_map_state.player_territory_pixels,
                    "done": done,
                }
            )

            if episode_step % 50 == 0:
                self.logger.info(
                    "Episode %s | step=%s | territory_ratio=%.4f | reward=%.3f | epsilon=%.4f",
                    episode_number,
                    episode_step,
                    current_map_state.player_territory_ratio,
                    reward,
                    self.agent.epsilon,
                )

            if self.debug_config.get("save_screenshots") and episode_step % int(
                self.debug_config.get("screenshot_every_n_steps", 100)
            ) == 0:
                self._save_debug_overlay(screenshot, episode_number, episode_step)

            if done:
                self.logger.info("Episode %s ended by defeat at step %s", episode_number, episode_step)
                break

        if done:
            self.game_launcher.close_defeat_popup()
        else:
            self._exit_to_menu()

        self.agent.complete_episode(total_reward, current_map_state.player_territory_ratio, decay=False)
        if self.training_enabled:
            self.agent.decay_epsilon()

        if self.training_enabled and self.agent.episode_count % self.agent_config["q_table_save_interval"] == 0:
            self.save_q_table()

        stats = {
            "episode": self.agent.episode_count,
            "worker_id": self.worker_id,
            "steps": episode_step,
            "total_steps": episode_step,
            "total_reward": total_reward,
            "final_territory_ratio": current_map_state.player_territory_ratio,
            "final_territory_pixels": current_map_state.player_territory_pixels,
            "epsilon": self.agent.epsilon,
            "player_color_hsv": current_map_state.player_color_hsv,
            "duration_seconds": timer.elapsed_seconds(),
            "won": False,
        }
        self.episode_logger.log_episode(stats)
        return stats

    def run(self, num_episodes: int) -> list[dict[str, Any]]:
        """
        Run multiple episodes with crash recovery.

        Args:
            num_episodes (int): Number of episodes to run.

        Returns:
            list[dict[str, Any]]: Completed episode summaries.
        """

        episode_stats: list[dict[str, Any]] = []
        for _ in range(num_episodes):
            try:
                stats = self.run_episode()
                if not stats:
                    continue
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
