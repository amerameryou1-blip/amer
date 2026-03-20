# Territorial.io Q-Learning Bot

This project is a full Python automation and reinforcement-learning bot for [Territorial.io](https://territorial.io). It uses Playwright to drive a real Chromium browser, OpenCV to interpret game screenshots, and a sparse tabular Q-learning agent to learn policies across many games.

## Features

- Real browser automation with Playwright sync API
- Screenshot-based map parsing with OpenCV
- Sparse numpy-backed Q-table with checkpoint save/load
- Single-worker and multi-worker training flows
- CSV episode logging and structured file/console logs
- Kaggle-ready notebook for quick bootstrapping

## Project Layout

- `main.py`: CLI entry point for training, evaluation, watch mode, and reset operations
- `config.json`: Human-editable runtime configuration
- `config_schema.md`: Explanation of every configuration field
- `browser/`: Playwright lifecycle, menu navigation, and input control
- `vision/`: Screenshot capture, color masks, map parsing, and state encoding
- `agent/`: Action space, Q-table, Q-learning agent, and rewards
- `training/`: Episode loop, multiprocessing coordinator, and CSV logging
- `utils/`: Logging, timers, and file safety helpers
- `kaggle_runner.ipynb`: Kaggle notebook for setup and checkpoint export

## Local Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install playwright opencv-python pillow numpy
playwright install chromium
```

3. Move into the project directory:

```bash
cd territorial_bot
```

## Start Training

Single worker:

```bash
python main.py --mode train --workers 1
```

Multiple workers:

```bash
python main.py --mode train --workers 4
```

Custom episode count:

```bash
python main.py --mode train --workers 2 --episodes 500
```

## Resume Training

Training resumes automatically when `training.resume_training` is `true` and the checkpoint at `agent.q_table_path` exists. Worker checkpoints are stored in the configured checkpoint directory and merged back into the primary Q-table during parallel training.

## Evaluate a Trained Bot

```bash
python main.py --mode eval --episodes 10
```

Evaluation loads the saved Q-table and forces a greedy policy when a checkpoint is available.

## Watch Mode

```bash
python main.py --mode watch
```

Watch mode forces a visible browser window and runs evaluation logic without training updates.

## Reset the Q-Table

```bash
python main.py --reset-qtable
```

## Kaggle Usage

Open [`kaggle_runner.ipynb`](./kaggle_runner.ipynb) in Kaggle, upload the project files into `/kaggle/working/territorial_bot`, and run the notebook cells in order. The notebook installs Playwright Chromium, applies Kaggle-specific config overrides, starts training, and exposes a downloadable checkpoint link.

## Config Reference

Configuration details are documented in [`config_schema.md`](./config_schema.md).

## Known Issues And Limitations

- Territorial.io UI selectors and post-match labels can change over time; `browser/game_launcher.py` uses layered heuristics and may need retuning if the site layout changes.
- Territory color segmentation is based on static HSV profiles. If the in-game palette differs from the defaults, update `vision/color_profiles.py` or the map crop configuration.
- Browser game automation can be sensitive to latency, ads, and unexpected popups. The browser manager and trainer recover from many failures, but long unattended runs still benefit from periodic checkpoint review.
