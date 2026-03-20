"""
Parallel training coordinator for Territorial.io bot.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

from agent.action_space import ActionSpace
from agent.q_agent import QAgent
from agent.q_table import QTable
from agent.reward import RewardCalculator
from browser.browser_manager import BrowserManager
from browser.game_controller import GameController
from browser.game_launcher import GameLauncher
from training.episode_logger import EpisodeLogger
from training.trainer import Trainer
from utils.file_utils import ensure_dir, list_worker_checkpoints
from utils.logger import setup_logger
from vision.map_parser import MapParser
from vision.state_encoder import StateEncoder


def worker_process_main(
    config: dict[str, Any],
    worker_id: int,
    assigned_episodes: int,
    initial_completed: int,
    overall_target: int,
    status_dict: Any,
    csv_lock: Any,
    stop_event: Any,
) -> None:
    """
    Entry point for a single parallel training worker process.

    Args:
        config (dict[str, Any]): Project configuration dictionary.
        worker_id (int): Worker identifier.
        assigned_episodes (int): Number of episodes assigned to this worker.
        initial_completed (int): Completed-episode count carried across restarts.
        overall_target (int): Total episode target originally assigned to the worker.
        status_dict (Any): Shared status dictionary.
        csv_lock (Any): Shared CSV write lock.
        stop_event (Any): Shared shutdown event.
    """

    worker_config = copy.deepcopy(config)
    checkpoint_dir = Path(worker_config["training"]["checkpoint_dir"])
    merged_qtable_path = Path(worker_config["agent"]["q_table_path"])
    worker_qtable_path = checkpoint_dir / f"worker_{worker_id}_q_table.pkl"
    worker_config["agent"]["q_table_path"] = str(worker_qtable_path)

    logger = setup_logger(f"worker_{worker_id}", worker_config["training"]["log_dir"])
    action_space = ActionSpace(worker_config)
    state_encoder = StateEncoder(worker_config)
    q_table = QTable(state_encoder.state_space_size(), action_space.get_action_count(), logger=logger)

    if worker_config["training"]["resume_training"]:
        if not q_table.load(merged_qtable_path):
            q_table.load(worker_qtable_path)

    agent = QAgent(worker_config, q_table, action_space, logger=logger)
    browser_manager = BrowserManager(worker_config, logger=logger)
    page = browser_manager.launch()
    game_controller = GameController(page, worker_config, logger=logger)
    game_launcher = GameLauncher(browser_manager, worker_config, logger=logger)
    map_parser = MapParser(worker_config, logger=logger)
    reward_calculator = RewardCalculator(worker_config["rewards"], logger=logger)
    episode_logger = EpisodeLogger(worker_config, worker_id=worker_id, write_lock=csv_lock, logger=logger)
    trainer = Trainer(
        worker_config,
        agent,
        browser_manager,
        game_launcher,
        game_controller,
        map_parser,
        state_encoder,
        reward_calculator,
        episode_logger,
        worker_id=worker_id,
        training_enabled=True,
        logger=logger,
    )

    reward_window: list[float] = []
    sync_interval = worker_config["training"]["worker_sync_interval"]

    status_dict[worker_id] = {
        "pid": os.getpid(),
        "episodes_completed": initial_completed,
        "target_episodes": overall_target,
        "avg_reward": 0.0,
        "last_reward": 0.0,
        "state": "running",
    }

    try:
        for episode_index in range(assigned_episodes):
            if stop_event.is_set():
                break

            stats = trainer.run_episode()
            reward_window.append(stats["total_reward"])
            reward_window = reward_window[-20:]
            avg_reward = sum(reward_window) / max(len(reward_window), 1)
            completed_total = initial_completed + episode_index + 1
            status_dict[worker_id] = {
                "pid": os.getpid(),
                "episodes_completed": completed_total,
                "target_episodes": overall_target,
                "avg_reward": avg_reward,
                "last_reward": stats["total_reward"],
                "state": "running",
            }

            if (episode_index + 1) % sync_interval == 0 or (episode_index + 1) == assigned_episodes:
                trainer.save_q_table(worker_qtable_path)
                if merged_qtable_path.exists():
                    q_table.load(merged_qtable_path)

        trainer.save_q_table(worker_qtable_path)
        status = dict(status_dict.get(worker_id, {}))
        status["state"] = "completed"
        status_dict[worker_id] = status
    except KeyboardInterrupt:
        trainer.save_q_table(worker_qtable_path)
        status = dict(status_dict.get(worker_id, {}))
        status["state"] = "interrupted"
        status_dict[worker_id] = status
    except Exception as exc:
        logger.exception("Worker %s crashed: %s", worker_id, exc)
        trainer.save_q_table(worker_qtable_path)
        status = dict(status_dict.get(worker_id, {}))
        status["state"] = "crashed"
        status_dict[worker_id] = status
        raise
    finally:
        browser_manager.close()


class ParallelTrainer:
    """
    Coordinate multiple browser workers and periodically merge their Q-tables.
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize the parallel trainer coordinator.

        Args:
            config (dict[str, Any]): Project configuration dictionary.
            logger (logging.Logger | None): Optional logger instance.
        """

        self.config = config
        self.logger = logger or logging.getLogger("parallel_trainer")
        self.checkpoint_dir = ensure_dir(config["training"]["checkpoint_dir"])
        self.log_dir = ensure_dir(config["training"]["log_dir"])
        self.merged_qtable_path = Path(config["agent"]["q_table_path"])
        self.processes: dict[int, multiprocessing.Process] = {}
        self.worker_targets: dict[int, int] = {}
        self.num_workers = 0
        self.manager: Any | None = None
        self.status_dict: Any | None = None
        self.csv_lock: Any | None = None
        self.stop_event: Any | None = None
        self.action_space_size = ActionSpace(config).get_action_count()
        self.state_space_size = StateEncoder(config).state_space_size()

    def _distribute_episodes(self, total_episodes: int, num_workers: int) -> dict[int, int]:
        """
        Split episode assignments as evenly as possible across workers.

        Args:
            total_episodes (int): Total requested episode count.
            num_workers (int): Number of worker processes.

        Returns:
            dict[int, int]: Per-worker episode targets.
        """

        base = total_episodes // max(num_workers, 1)
        remainder = total_episodes % max(num_workers, 1)
        targets: dict[int, int] = {}
        for worker_id in range(num_workers):
            targets[worker_id] = base + (1 if worker_id < remainder else 0)
        return targets

    def _spawn_worker(self, worker_id: int, assigned_episodes: int, initial_completed: int = 0) -> None:
        """
        Spawn a worker process for a specific episode allocation.

        Args:
            worker_id (int): Worker identifier.
            assigned_episodes (int): Number of episodes to run.
            initial_completed (int): Previously completed episodes for this worker.
        """

        if assigned_episodes <= 0 or self.status_dict is None or self.csv_lock is None or self.stop_event is None:
            return

        process = multiprocessing.Process(
            target=worker_process_main,
            args=(
                self.config,
                worker_id,
                assigned_episodes,
                initial_completed,
                self.worker_targets.get(worker_id, assigned_episodes),
                self.status_dict,
                self.csv_lock,
                self.stop_event,
            ),
            name=f"territorial_worker_{worker_id}",
        )
        process.start()
        self.processes[worker_id] = process
        self.logger.info("Spawned worker %s with PID %s for %s episodes", worker_id, process.pid, assigned_episodes)

    def _merge_worker_tables(self) -> bool:
        """
        Merge available worker checkpoints into the shared Q-table checkpoint.

        Returns:
            bool: True when at least one worker checkpoint was merged.
        """

        checkpoints = list_worker_checkpoints(self.checkpoint_dir, self.num_workers)
        if not checkpoints:
            return False

        merged_table: QTable | None = None
        for checkpoint_path in checkpoints:
            worker_table = QTable(self.state_space_size, self.action_space_size, logger=self.logger)
            if not worker_table.load(checkpoint_path):
                continue
            if merged_table is None:
                merged_table = worker_table
            else:
                merged_table.merge(worker_table)

        if merged_table is None:
            return False

        merged_table.save(self.merged_qtable_path)
        self.logger.info("Merged %s worker checkpoints into %s", len(checkpoints), self.merged_qtable_path)
        return True

    def _restart_dead_workers(self) -> None:
        """
        Restart crashed workers that still have remaining assigned episodes.
        """

        if self.status_dict is None:
            return

        for worker_id, process in list(self.processes.items()):
            if process.is_alive():
                continue

            status = dict(self.status_dict.get(worker_id, {}))
            completed = int(status.get("episodes_completed", 0))
            target = self.worker_targets.get(worker_id, 0)
            remaining = target - completed
            if remaining <= 0:
                continue

            self.logger.warning(
                "Worker %s died unexpectedly; restarting with %s remaining episodes",
                worker_id,
                remaining,
            )
            self._spawn_worker(worker_id, remaining, initial_completed=completed)

    def _all_workers_finished(self) -> bool:
        """
        Check whether all assigned worker episodes have completed.

        Returns:
            bool: True when all workers reached their targets.
        """

        if self.status_dict is None:
            return True

        for worker_id, target in self.worker_targets.items():
            status = dict(self.status_dict.get(worker_id, {}))
            completed = int(status.get("episodes_completed", 0))
            if completed < target:
                return False
        return True

    def _log_status_table(self) -> None:
        """
        Log a compact live table of per-worker status.
        """

        if self.status_dict is None:
            return

        lines = ["Worker | PID | State | Episodes | Target | AvgReward | LastReward"]
        for worker_id in range(self.num_workers):
            status = dict(self.status_dict.get(worker_id, {}))
            lines.append(
                f"{worker_id} | {status.get('pid', '-')}"
                f" | {status.get('state', 'pending')}"
                f" | {status.get('episodes_completed', 0)}"
                f" | {status.get('target_episodes', self.worker_targets.get(worker_id, 0))}"
                f" | {float(status.get('avg_reward', 0.0)):.3f}"
                f" | {float(status.get('last_reward', 0.0)):.3f}"
            )
        self.logger.info("Parallel status:\n%s", "\n".join(lines))

    def start(self, num_workers: int, total_episodes: int | None = None) -> None:
        """
        Start worker processes for parallel training.

        Args:
            num_workers (int): Number of worker processes.
            total_episodes (int | None): Optional total episode budget.
        """

        self.num_workers = num_workers
        self.manager = multiprocessing.Manager()
        self.status_dict = self.manager.dict()
        self.csv_lock = multiprocessing.Lock()
        self.stop_event = multiprocessing.Event()
        total = total_episodes or self.config["game"]["max_episodes"]
        self.worker_targets = self._distribute_episodes(total, num_workers)
        for worker_id, assigned in self.worker_targets.items():
            self._spawn_worker(worker_id, assigned)

    def monitor(self) -> None:
        """
        Monitor workers, merge checkpoints, and restart dead workers as needed.
        """

        if self.stop_event is None:
            raise RuntimeError("ParallelTrainer.start() must be called before monitor()")

        last_report = 0.0
        last_merge = 0.0
        while not self.stop_event.is_set():
            now = time.time()
            self._restart_dead_workers()

            if now - last_merge >= 10.0:
                self._merge_worker_tables()
                last_merge = now

            if now - last_report >= 30.0:
                self._log_status_table()
                last_report = now

            if self._all_workers_finished():
                break
            time.sleep(5.0)

        self._merge_worker_tables()

    def stop(self) -> None:
        """
        Gracefully stop all workers, join processes, and persist the merged checkpoint.
        """

        if self.stop_event is not None:
            self.stop_event.set()

        for process in self.processes.values():
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

        self._merge_worker_tables()
        if self.manager is not None:
            self.manager.shutdown()
