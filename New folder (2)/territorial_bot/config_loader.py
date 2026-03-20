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
        "multiplayer_mode": bool,
    },
    "vision": {
        "map_region": list,
        "territory_sample_grid": int,
        "color_tolerance": int,
        "min_territory_pixels": int,
        "use_grayscale_fallback": bool,
        "process_width": int,
        "process_height": int,
        "playable_area": list,
        "troop_bar_sample_pixel": list,
        "defeat_check_region": list,
        "leaderboard_region": list,
        "stats_region": list,
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
        "grid_size": int,
        "directional_offset_px": int,
        "hold_duration_ms": int,
        "action_delay_ms": int,
        "playable_x_start": int,
        "playable_x_end": int,
        "playable_y_start": int,
        "playable_y_end": int,
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
        "top_half_leaderboard_bonus": float,
    },
    "debug": {
        "save_screenshots": bool,
        "screenshot_every_n_steps": int,
        "verbose_logging": bool,
    },
}


def _validate_schema(config: dict[str, Any], schema: dict[str, Any], prefix: str = "") -> None:
    """
    Recursively validate required configuration fields and their basic types.

    Args:
        config (dict[str, Any]): Parsed configuration dictionary.
        schema (dict[str, Any]): Required schema definition.
        prefix (str): Dot-separated path prefix for nested fields.

    Raises:
        ConfigError: If a required field is missing or has the wrong type.
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

        if expected is float and (not isinstance(value, (float, int)) or isinstance(value, bool)):
            raise ConfigError(f"Config field '{full_key}' must be a number")
        if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
            raise ConfigError(f"Config field '{full_key}' must be an integer")
        if expected is bool and not isinstance(value, bool):
            raise ConfigError(f"Config field '{full_key}' must be a boolean")
        if expected is str and not isinstance(value, str):
            raise ConfigError(f"Config field '{full_key}' must be a string")
        if expected is list and not isinstance(value, list):
            raise ConfigError(f"Config field '{full_key}' must be a list")

    list_fields = {
        "vision.map_region": 4,
        "vision.playable_area": 4,
        "vision.troop_bar_sample_pixel": 2,
        "vision.defeat_check_region": 4,
        "vision.leaderboard_region": 4,
        "vision.stats_region": 4,
    }
    for field_name, expected_length in list_fields.items():
        section, field = field_name.split(".", 1)
        values = config[section][field]
        if len(values) != expected_length:
            raise ConfigError(f"Config field '{field_name}' must have length {expected_length}")

    if config["agent"]["epsilon_decay_strategy"] not in {"exponential", "linear"}:
        raise ConfigError("Config field 'agent.epsilon_decay_strategy' must be 'exponential' or 'linear'")

    if config["actions"]["grid_size"] <= 0:
        raise ConfigError("Config field 'actions.grid_size' must be greater than 0")

    if config["agent"]["state_bins"] <= 0:
        raise ConfigError("Config field 'agent.state_bins' must be greater than 0")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and validate the project configuration file.

    Args:
        config_path (str | Path | None): Optional path to a config file.

    Returns:
        dict[str, Any]: Validated configuration dictionary.

    Raises:
        ConfigError: If reading, parsing, or validation fails.
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
