"""
Configuration loading and validation utilities for Territorial.io bot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """
    Raised when the project configuration is missing required fields or is invalid.
    """


REQUIRED_CONFIG_SCHEMA = {
    "game": {
        "url": str,
        "player_name": str,
        "headless": bool,
        "browser_timeout_ms": int,
        "game_start_wait_s": int,
        "screenshot_interval_ms": int,
        "max_episode_steps": int,
        "max_episodes": int,
        "restart_on_death": bool,
        "window_width": int,
        "window_height": int,
    },
    "vision": {
        "map_region": list,
        "territory_sample_grid": int,
        "color_tolerance": int,
        "min_territory_pixels": int,
        "use_grayscale_fallback": bool,
    },
    "agent": {
        "learning_rate": float,
        "discount_factor": float,
        "epsilon_start": float,
        "epsilon_min": float,
        "epsilon_decay": float,
        "epsilon_decay_strategy": str,
        "state_bins": int,
        "q_table_save_interval": int,
        "q_table_path": str,
    },
    "actions": {
        "move_directions": list,
        "click_grid_size": int,
        "hold_duration_ms": int,
        "action_delay_ms": int,
    },
    "training": {
        "num_workers": int,
        "worker_sync_interval": int,
        "log_dir": str,
        "checkpoint_dir": str,
        "resume_training": bool,
        "eval_every_n_episodes": int,
        "eval_episodes": int,
    },
    "rewards": {
        "territory_gain_reward": float,
        "territory_loss_penalty": float,
        "death_penalty": float,
        "survival_bonus_per_step": float,
        "win_bonus": float,
        "idle_penalty": float,
        "attacking_enemy_bonus": float,
        "neutral_capture_reward": float,
    },
}


def _validate_schema(config: dict[str, Any], schema: dict[str, Any], prefix: str = "") -> None:
    """
    Recursively validate required configuration fields and lightweight types.

    Args:
        config (dict[str, Any]): Parsed configuration dictionary.
        schema (dict[str, Any]): Required schema definition.
        prefix (str): Dot-separated path prefix for nested fields.

    Raises:
        ConfigError: If a required field is missing or its type is invalid.
    """

    for key, expected in schema.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in config:
            raise ConfigError(f"Missing required config field: {full_key}")

        value = config[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"Config field '{full_key}' must be an object")
            _validate_schema(value, expected, full_key)
            continue

        if expected is float and not isinstance(value, (float, int)):
            raise ConfigError(f"Config field '{full_key}' must be a number")
        elif expected is int and not isinstance(value, int):
            raise ConfigError(f"Config field '{full_key}' must be an integer")
        elif expected is bool and not isinstance(value, bool):
            raise ConfigError(f"Config field '{full_key}' must be a boolean")
        elif expected is str and not isinstance(value, str):
            raise ConfigError(f"Config field '{full_key}' must be a string")
        elif expected is list and not isinstance(value, list):
            raise ConfigError(f"Config field '{full_key}' must be a list")

    map_region = config["vision"]["map_region"]
    if len(map_region) != 4:
        raise ConfigError("Config field 'vision.map_region' must contain [x, y, width, height]")

    if config["agent"]["epsilon_decay_strategy"] not in {"exponential", "linear"}:
        raise ConfigError(
            "Config field 'agent.epsilon_decay_strategy' must be 'exponential' or 'linear'"
        )

    if len(config["actions"]["move_directions"]) != 8:
        raise ConfigError("Config field 'actions.move_directions' must define exactly 8 directions")

    if config["actions"]["click_grid_size"] <= 0:
        raise ConfigError("Config field 'actions.click_grid_size' must be greater than 0")

    if config["agent"]["state_bins"] <= 0:
        raise ConfigError("Config field 'agent.state_bins' must be greater than 0")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and validate the project configuration JSON file.

    Args:
        config_path (str | Path | None): Optional path to a config file.

    Returns:
        dict[str, Any]: Validated configuration dictionary.

    Raises:
        ConfigError: If the file cannot be read, parsed, or validated.
    """

    resolved_path = Path(config_path) if config_path else Path(__file__).resolve().parent / "config.json"

    if not resolved_path.exists():
        raise ConfigError(f"Config file not found: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file '{resolved_path}': {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read config file '{resolved_path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError("Root config document must be a JSON object")

    _validate_schema(config, REQUIRED_CONFIG_SCHEMA)
    return config
