# Config Schema

This document explains every field in [`config.json`](./config.json).

## `game`

- `url`: Browser URL to open for Territorial.io.
- `player_name`: In-game name typed into the menu before match start.
- `headless`: Launch Chromium without a visible window when `true`.
- `browser_timeout_ms`: Default Playwright timeout for navigation and selectors.
- `game_start_wait_s`: Delay after pressing play before the first observation step.
- `screenshot_interval_ms`: Delay between action execution and the next screenshot.
- `max_episode_steps`: Maximum environment steps per episode before forced stop.
- `max_episodes`: Default total number of episodes for a training run.
- `restart_on_death`: Restart the browser session when a match ends or crashes.
- `window_width`: Browser viewport width in pixels.
- `window_height`: Browser viewport height in pixels.

## `vision`

- `map_region`: `[x, y, width, height]` crop for the map area to analyze.
- `territory_sample_grid`: Reserved sampling granularity for grid-based map inspection.
- `color_tolerance`: Extra HSV tolerance used by mask generation and fallbacks.
- `min_territory_pixels`: Minimum player-pixel threshold before centroid logic trusts a detection.
- `use_grayscale_fallback`: Enables grayscale fallback heuristics when color segmentation is weak.

## `agent`

- `learning_rate`: Q-learning alpha value.
- `discount_factor`: Q-learning gamma value.
- `epsilon_start`: Initial exploration probability.
- `epsilon_min`: Minimum exploration floor after decay.
- `epsilon_decay`: Decay factor or step size depending on strategy.
- `epsilon_decay_strategy`: `"exponential"` or `"linear"`.
- `state_bins`: Discretization count used by the state encoder.
- `q_table_save_interval`: Save Q-table after this many episodes.
- `q_table_path`: Primary checkpoint path for the merged Q-table.

## `actions`

- `move_directions`: Ordered list of 8 directional nudge actions.
- `click_grid_size`: Grid dimension for map click actions. `8` means `8 x 8`.
- `hold_duration_ms`: Default mouse hold duration for attack clicks.
- `action_delay_ms`: Cooldown inserted after every action.

## `training`

- `num_workers`: Default worker count for parallel training.
- `worker_sync_interval`: Episodes between worker checkpoint syncs.
- `log_dir`: Directory for logs, CSVs, and debug output.
- `checkpoint_dir`: Directory for worker and merged Q-table checkpoints.
- `resume_training`: Load existing checkpoints when available.
- `eval_every_n_episodes`: Reserved interval for scheduled evaluation hooks.
- `eval_episodes`: Default number of evaluation episodes.

## `rewards`

- `territory_gain_reward`: Reward multiplier applied to gained territory pixels.
- `territory_loss_penalty`: Penalty multiplier applied to lost territory pixels.
- `death_penalty`: Terminal penalty for losing a match.
- `survival_bonus_per_step`: Small positive reward for surviving each step.
- `win_bonus`: Terminal bonus for winning a match.
- `idle_penalty`: Penalty applied after prolonged stagnant territory.
- `attacking_enemy_bonus`: Bonus when growth occurs while moving closer to enemies.
- `neutral_capture_reward`: Bonus when player growth coincides with neutral capture.
