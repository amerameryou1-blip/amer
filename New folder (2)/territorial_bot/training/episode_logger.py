"""
Episode-level logging utilities for Territorial.io bot.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from utils.file_utils import ensure_dir


class EpisodeLogger:
    """
    Log episode summaries to both CSV and the configured logger.
    """

    def __init__(
        self,
        config: dict[str, Any],
        worker_id: int = 0,
        write_lock: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize episode logging targets and rolling statistics storage.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            worker_id (int): Worker identifier for multi-process runs.
            write_lock (Any | None): Optional multiprocessing lock for CSV writes.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.config = config
        self.worker_id = worker_id
        self.write_lock = write_lock
        self.logger = logger or logging.getLogger("episode_logger")
        self.log_dir = ensure_dir(config["training"]["log_dir"])
        self.csv_path = Path(self.log_dir) / "episodes.csv"
        self.debug_mode = bool(config.get("debug", False) or config["training"].get("debug_step_logging", False))
        self.rolling_window: list[dict[str, Any]] = []
        self.headers = [
            "episode",
            "worker_id",
            "total_steps",
            "total_reward",
            "final_territory_ratio",
            "epsilon",
            "duration_seconds",
            "won",
        ]
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """
        Create the episode CSV file and header row if needed.
        """

        lock = self.write_lock
        if lock is not None:
            lock.acquire()
        try:
            if self.csv_path.exists():
                return
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.headers)
                writer.writeheader()
        finally:
            if lock is not None:
                lock.release()

    def log_episode(self, stats_dict: dict[str, Any]) -> None:
        """
        Append an episode summary to the CSV file and log a concise summary line.

        Args:
            stats_dict (dict[str, Any]): Episode summary fields.
        """

        row = {
            "episode": stats_dict["episode"],
            "worker_id": stats_dict.get("worker_id", self.worker_id),
            "total_steps": stats_dict["total_steps"],
            "total_reward": stats_dict["total_reward"],
            "final_territory_ratio": stats_dict["final_territory_ratio"],
            "epsilon": stats_dict["epsilon"],
            "duration_seconds": stats_dict["duration_seconds"],
            "won": stats_dict["won"],
        }

        lock = self.write_lock
        if lock is not None:
            lock.acquire()
        try:
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.headers)
                writer.writerow(row)
        finally:
            if lock is not None:
                lock.release()

        self.rolling_window.append(row)
        self.rolling_window = self.rolling_window[-100:]
        self.logger.info(
            "Episode %s | worker=%s | steps=%s | reward=%.3f | ratio=%.4f | epsilon=%.4f | won=%s",
            row["episode"],
            row["worker_id"],
            row["total_steps"],
            row["total_reward"],
            row["final_territory_ratio"],
            row["epsilon"],
            row["won"],
        )

    def log_step(self, step_data: dict[str, Any]) -> None:
        """
        Optionally log verbose per-step details when debug mode is enabled.

        Args:
            step_data (dict[str, Any]): Step-level diagnostic data.
        """

        if not self.debug_mode:
            return
        self.logger.debug("Step data: %s", step_data)

    def print_training_summary(self, n_episodes: int) -> None:
        """
        Log rolling average statistics for the most recent episodes.

        Args:
            n_episodes (int): Number of recent episodes to summarize.
        """

        if not self.rolling_window:
            self.logger.info("No episode data available for summary")
            return

        window = self.rolling_window[-n_episodes:]
        avg_reward = sum(item["total_reward"] for item in window) / max(len(window), 1)
        avg_steps = sum(item["total_steps"] for item in window) / max(len(window), 1)
        avg_ratio = sum(item["final_territory_ratio"] for item in window) / max(len(window), 1)
        win_rate = sum(1 for item in window if item["won"]) / max(len(window), 1)
        self.logger.info(
            "Last %s episodes | avg_reward=%.3f | avg_steps=%.1f | avg_ratio=%.4f | win_rate=%.2f%%",
            len(window),
            avg_reward,
            avg_steps,
            avg_ratio,
            win_rate * 100.0,
        )
