"""
Command-line entry point for training, evaluation, and watch modes.
"""

from __future__ import annotations

import argparse
import copy
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
    Handle SIGINT and SIGTERM by saving state before stopping the process.

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
    Resolve relative log and checkpoint paths against the project directory.

    Args:
        config (dict[str, Any]): Loaded configuration dictionary.
        base_dir (Path): Project root directory.

    Returns:
        dict[str, Any]: Path-normalized configuration dictionary.
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


def _build_single_trainer(
    config: dict[str, Any],
    worker_id: int = 0,
    training_enabled: bool = True,
    logger_name: str = "main",
    load_checkpoint: bool = True,
) -> tuple[Trainer, logging.Logger, bool]:
    """
    Build a fully wired single-worker trainer instance.

    Args:
        config (dict[str, Any]): Runtime configuration dictionary.
        worker_id (int): Worker identifier.
        training_enabled (bool): Whether this trainer should update Q-values.
        logger_name (str): Logger namespace.
        load_checkpoint (bool): Whether to try loading the primary checkpoint.

    Returns:
        tuple[Trainer, logging.Logger, bool]: Trainer, shared logger, and checkpoint-loaded flag.
    """

    logger = setup_logger(logger_name, config["training"]["log_dir"])
    action_space = ActionSpace(config)
    state_encoder = StateEncoder(config)
    q_table = QTable(state_encoder.state_space_size(), action_space.get_action_count(), logger=logger)
    checkpoint_loaded = False
    if load_checkpoint:
        checkpoint_loaded = q_table.load(config["agent"]["q_table_path"])

    agent = QAgent(config, q_table, action_space, logger=logger)
    browser_manager = BrowserManager(config, logger=logger)
    page = browser_manager.launch()
    game_controller = GameController(page, config, logger=logger)
    game_launcher = GameLauncher(browser_manager, config, logger=logger)
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
    Delete the primary merged Q-table checkpoint and exit.

    Args:
        config_path (str | Path | None): Optional config file path.

    Returns:
        bool: True when a checkpoint file was removed.
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
        if checkpoint.exists():
            checkpoint.unlink()
            removed_paths.append(checkpoint)

    if not removed_paths:
        logger.info("No Q-table checkpoints found in %s", checkpoint_dir)
        return False

    logger.info("Deleted %s checkpoint file(s)", len(removed_paths))
    return True


def run_training(workers: int = 1, episodes: int | None = None, config_path: str | Path | None = None) -> Any:
    """
    Run training in either single-worker or parallel mode.

    Args:
        workers (int): Number of training workers.
        episodes (int | None): Optional episode count override.
        config_path (str | Path | None): Optional config file path.

    Returns:
        Any: Training results summary.
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
    logger.info("Single-worker training started | episodes=%s | checkpoint_loaded=%s", total_episodes, checkpoint_loaded)
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
    Run evaluation episodes without updating the Q-table.

    Args:
        episodes (int): Number of evaluation episodes to run.
        config_path (str | Path | None): Optional config file path.
        headed (bool): Whether to force a visible browser window.

    Returns:
        list[dict[str, Any]]: Episode summaries from evaluation.
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

    if checkpoint_loaded:
        trainer.agent.epsilon = 0.0
    else:
        logger.warning("No checkpoint loaded for evaluation; falling back to exploratory policy")
        trainer.agent.epsilon = 1.0

    try:
        stats = trainer.run(episodes)
        wins = sum(1 for item in stats if item["won"])
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


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Territorial.io Q-learning bot")
    parser.add_argument("--mode", choices=["train", "eval", "watch"], default="train")
    parser.add_argument("--workers", type=int, default=None, help="Number of training workers")
    parser.add_argument("--episodes", type=int, default=None, help="Episode override")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--reset-qtable", action="store_true", help="Delete the saved Q-table and exit")
    return parser


def main() -> int:
    """
    Parse CLI arguments and run the requested Territorial.io workflow.

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
