"""
CLI entry point for Territorial.io training, evaluation, and vision debugging.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from agent.action_space import ActionSpace
from agent.q_agent import QAgent
from agent.q_table import QTable
from agent.reward import RewardCalculator
from browser.browser_manager import BrowserManager
from browser.game_controller import GameController
from browser.game_launcher import GameLauncher
from config_loader import ConfigError, load_config
from training.episode_logger import EpisodeLogger
from training.parallel_trainer import ParallelTrainer
from training.trainer import Trainer
from utils.file_utils import ensure_dir
from utils.logger import setup_logger
from vision.map_parser import MapParser
from vision.state_encoder import StateEncoder


ACTIVE_RUNTIME: Any | None = None


def _signal_handler(signum: int, frame: Any) -> None:
    """
    Save state and stop cleanly on process signals.

    Args:
        signum (int): Signal number.
        frame (Any): Current stack frame.
    """

    del frame
    if ACTIVE_RUNTIME is not None:
        if hasattr(ACTIVE_RUNTIME, "save_q_table"):
            ACTIVE_RUNTIME.save_q_table()
        if hasattr(ACTIVE_RUNTIME, "stop"):
            ACTIVE_RUNTIME.stop()
    raise KeyboardInterrupt(f"Received signal {signum}")


def _resolve_runtime_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """
    Resolve runtime paths against the project directory.

    Args:
        config (dict[str, Any]): Configuration dictionary.
        base_dir (Path): Project root directory.

    Returns:
        dict[str, Any]: Configuration dictionary with resolved filesystem paths.
    """

    resolved = copy.deepcopy(config)
    log_dir = Path(resolved["training"]["log_dir"])
    checkpoint_dir = Path(resolved["training"]["checkpoint_dir"])
    q_table_path = Path(resolved["agent"]["q_table_path"])

    if not log_dir.is_absolute():
        log_dir = base_dir / log_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = base_dir / checkpoint_dir
    if not q_table_path.is_absolute():
        q_table_path = base_dir / q_table_path

    ensure_dir(log_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(q_table_path.parent)

    resolved["training"]["log_dir"] = str(log_dir)
    resolved["training"]["checkpoint_dir"] = str(checkpoint_dir)
    resolved["agent"]["q_table_path"] = str(q_table_path)
    return resolved


def _read_resume_episode(log_dir: str | Path) -> int:
    """
    Read the last completed episode number from `episodes.csv` when present.

    Args:
        log_dir (str | Path): Log directory.

    Returns:
        int: Last completed episode number, or 0 if unavailable.
    """

    csv_path = Path(log_dir) / "episodes.csv"
    if not csv_path.exists():
        return 0

    last_episode = 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("episode"):
                    last_episode = max(last_episode, int(float(row["episode"])))
    except Exception:
        return 0
    return last_episode


def _build_single_trainer(
    config: dict[str, Any],
    worker_id: int = 0,
    training_enabled: bool = True,
    logger_name: str = "main",
    load_checkpoint: bool = True,
) -> tuple[Trainer, logging.Logger, bool]:
    """
    Construct a fully wired single-worker trainer.

    Args:
        config (dict[str, Any]): Runtime configuration.
        worker_id (int): Worker identifier.
        training_enabled (bool): Whether training updates should be applied.
        logger_name (str): Logger namespace.
        load_checkpoint (bool): Whether to load the primary Q-table checkpoint.

    Returns:
        tuple[Trainer, logging.Logger, bool]: Trainer instance, logger, and checkpoint-loaded flag.
    """

    logger = setup_logger(logger_name, config["training"]["log_dir"])
    action_space = ActionSpace(config)
    state_encoder = StateEncoder(config)
    q_table = QTable(state_encoder.state_space_size(), action_space.get_action_count(), logger=logger)
    checkpoint_loaded = False
    if load_checkpoint:
        checkpoint_loaded = q_table.load(config["agent"]["q_table_path"])

    agent = QAgent(config, q_table, action_space, logger=logger)
    agent.episode_count = _read_resume_episode(config["training"]["log_dir"])
    browser_manager = BrowserManager(config, logger=logger)
    page = browser_manager.launch()
    game_controller = GameController(page, config, logger=logger)
    game_launcher = GameLauncher(page, config, logger=logger)
    map_parser = MapParser(config, logger=logger)
    reward_calculator = RewardCalculator(config["rewards"], logger=logger)
    episode_logger = EpisodeLogger(config, worker_id=worker_id, logger=logger)
    trainer = Trainer(
        config,
        agent,
        browser_manager,
        game_launcher,
        game_controller,
        map_parser,
        state_encoder,
        reward_calculator,
        episode_logger,
        worker_id=worker_id,
        training_enabled=training_enabled,
        logger=logger,
    )
    return trainer, logger, checkpoint_loaded


def reset_qtable(config_path: str | Path | None = None) -> bool:
    """
    Delete stored Q-table checkpoints.

    Args:
        config_path (str | Path | None): Optional config path.

    Returns:
        bool: True when at least one checkpoint was deleted.
    """

    project_root = Path(__file__).resolve().parent
    config = _resolve_runtime_paths(load_config(config_path), project_root)
    logger = setup_logger("reset", config["training"]["log_dir"])
    q_table_path = Path(config["agent"]["q_table_path"])
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    removed_paths: list[Path] = []

    if q_table_path.exists():
        q_table_path.unlink()
        removed_paths.append(q_table_path)
    for checkpoint in checkpoint_dir.glob("*.pkl"):
        checkpoint.unlink()
        removed_paths.append(checkpoint)

    if not removed_paths:
        logger.info("No Q-table checkpoints found in %s", checkpoint_dir)
        return False

    logger.info("Deleted %s checkpoint file(s)", len(removed_paths))
    return True


def run_training(workers: int = 1, episodes: int | None = None, config_path: str | Path | None = None) -> Any:
    """
    Run Territorial.io training.

    Args:
        workers (int): Number of workers.
        episodes (int | None): Optional episode count override.
        config_path (str | Path | None): Optional config path.

    Returns:
        Any: Training results.
    """

    global ACTIVE_RUNTIME

    project_root = Path(__file__).resolve().parent
    config = _resolve_runtime_paths(load_config(config_path), project_root)
    total_episodes = episodes or config["game"]["max_episodes"]

    if workers > 1:
        logger = setup_logger("parallel", config["training"]["log_dir"])
        parallel_trainer = ParallelTrainer(config, logger=logger)
        ACTIVE_RUNTIME = parallel_trainer
        try:
            logger.info("Parallel training start | workers=%s | episodes=%s", workers, total_episodes)
            parallel_trainer.start(workers, total_episodes=total_episodes)
            parallel_trainer.monitor()
            return {"mode": "train", "workers": workers, "episodes": total_episodes}
        finally:
            parallel_trainer.stop()
            ACTIVE_RUNTIME = None

    trainer, logger, checkpoint_loaded = _build_single_trainer(
        config,
        training_enabled=True,
        logger_name="train",
        load_checkpoint=config["training"]["resume_training"],
    )
    ACTIVE_RUNTIME = trainer
    logger.info(
        "Single-worker training start | checkpoint_loaded=%s | resume_episode=%s | target_episodes=%s",
        checkpoint_loaded,
        trainer.agent.episode_count + 1,
        total_episodes,
    )
    try:
        stats = trainer.run(total_episodes)
        trainer.save_q_table()
        return stats
    finally:
        trainer.browser_manager.close()
        ACTIVE_RUNTIME = None


def run_evaluation(
    episodes: int = 10,
    config_path: str | Path | None = None,
    headed: bool = False,
) -> list[dict[str, Any]]:
    """
    Run evaluation episodes without Q-table updates.

    Args:
        episodes (int): Number of episodes.
        config_path (str | Path | None): Optional config path.
        headed (bool): Whether to force a visible browser.

    Returns:
        list[dict[str, Any]]: Evaluation episode summaries.
    """

    global ACTIVE_RUNTIME

    project_root = Path(__file__).resolve().parent
    config = _resolve_runtime_paths(load_config(config_path), project_root)
    if headed:
        config["game"]["headless"] = False

    trainer, logger, checkpoint_loaded = _build_single_trainer(
        config,
        training_enabled=False,
        logger_name="eval",
        load_checkpoint=True,
    )
    ACTIVE_RUNTIME = trainer
    trainer.agent.epsilon = 0.0 if checkpoint_loaded else 1.0
    if not checkpoint_loaded:
        logger.warning("No checkpoint found for evaluation; using exploratory policy")

    try:
        stats = trainer.run(episodes)
        wins = sum(1 for item in stats if item.get("won"))
        avg_reward = sum(item["total_reward"] for item in stats) / max(len(stats), 1)
        avg_ratio = sum(item["final_territory_ratio"] for item in stats) / max(len(stats), 1)
        logger.info(
            "Evaluation complete | episodes=%s | wins=%s | win_rate=%.2f%% | avg_reward=%.3f | avg_ratio=%.4f",
            len(stats),
            wins,
            (wins / max(len(stats), 1)) * 100.0,
            avg_reward,
            avg_ratio,
        )
        return stats
    finally:
        trainer.browser_manager.close()
        ACTIVE_RUNTIME = None


def run_debug_vision(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Run a single headed debug episode that saves overlay frames every 10 steps.

    Args:
        config_path (str | Path | None): Optional config path.

    Returns:
        list[dict[str, Any]]: One-item debug run result list.
    """

    global ACTIVE_RUNTIME

    project_root = Path(__file__).resolve().parent
    config = _resolve_runtime_paths(load_config(config_path), project_root)
    config["game"]["headless"] = False
    config["debug"]["save_screenshots"] = True
    config["debug"]["screenshot_every_n_steps"] = 10
    config["debug"]["verbose_logging"] = True

    trainer, logger, _ = _build_single_trainer(
        config,
        training_enabled=False,
        logger_name="debug_vision",
        load_checkpoint=False,
    )
    ACTIVE_RUNTIME = trainer
    trainer.agent.epsilon = 1.0
    logger.info("Vision debug run start | headed=%s | debug_frames_every=%s", True, 10)
    try:
        stats = trainer.run(1)
        return stats
    finally:
        trainer.browser_manager.close()
        ACTIVE_RUNTIME = None


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """

    parser = argparse.ArgumentParser(description="Territorial.io screenshot-grounded Q-learning bot")
    parser.add_argument("--mode", choices=["train", "eval", "watch"], default="train")
    parser.add_argument("--workers", type=int, default=None, help="Number of training workers")
    parser.add_argument("--episodes", type=int, default=None, help="Episode override")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--reset-qtable", action="store_true", help="Delete the saved Q-table and exit")
    parser.add_argument(
        "--debug-vision",
        action="store_true",
        help="Run one headed episode, save overlay frames every 10 steps, and exit",
    )
    return parser


def main() -> int:
    """
    Parse arguments and run the requested workflow.

    Returns:
        int: Process exit code.
    """

    parser = _build_parser()
    args = parser.parse_args()

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except ValueError:
        pass

    try:
        config = _resolve_runtime_paths(load_config(args.config), Path(__file__).resolve().parent)
        workers = args.workers or config["training"]["num_workers"]

        if args.reset_qtable:
            reset_qtable(args.config)
            return 0

        if args.debug_vision:
            run_debug_vision(args.config)
            return 0

        if args.mode == "train":
            run_training(workers=workers, episodes=args.episodes, config_path=args.config)
            return 0

        if args.mode == "eval":
            eval_episodes = args.episodes or config["training"]["eval_episodes"]
            run_evaluation(episodes=eval_episodes, config_path=args.config, headed=False)
            return 0

        watch_episodes = args.episodes or 1
        run_evaluation(episodes=watch_episodes, config_path=args.config, headed=True)
        return 0
    except ConfigError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
