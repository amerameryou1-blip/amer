#!/usr/bin/env python3

import functools
import importlib
import json
import math
import os
import random
import select
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote


GITHUB_PAT = ""
GITHUB_USERNAME = "amerameryou1-blip"
GITHUB_EMAIL = "you@example.com"


REPO_HTTPS_URL = "https://github.com/amerameryou1-blip/Mythos.git"
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
REPO_DIR = WORK_ROOT / "Mythos"
CYCLE_SECONDS = 3600
BOOTSTRAP_GAMES = 10
SELFPLAY_BUDGET_RATIO = 0.60
TRAINING_BUDGET_RATIO = 0.30
BOOTSTRAP_DEPTH = 4
CYCLE_DEPTH = 5
MAX_PLIES = 220
MAX_RECENT_SHARDS = 16
MAX_TRAINING_SAMPLES = 120_000
DEFAULT_HASH_MB = 128
TRAIN_THREADS = 1
SEED = 1337


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def redact(text: str) -> str:
    value = text
    if GITHUB_PAT.strip():
        value = value.replace(GITHUB_PAT, "***REDACTED_PAT***")
    auth_url = authenticated_repo_url() if "authenticated_repo_url" in globals() else ""
    if auth_url:
        value = value.replace(auth_url, REPO_HTTPS_URL)
    return value


def run(
    command: list[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = False,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    kwargs = {
        "cwd": str(cwd) if cwd is not None else None,
        "env": env,
        "text": True,
        "check": False,
    }
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    result = subprocess.run(command, **kwargs)
    if check and result.returncode != 0:
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(
            f"command failed ({result.returncode}): {redact(' '.join(command))}\n"
            f"stdout: {redact(stdout)}\nstderr: {redact(stderr)}"
        )
    return result


def ensure_python_package(import_name: str, pip_name: str) -> None:
    try:
        importlib.import_module(import_name)
    except Exception:
        log(f"Installing missing Python package: {pip_name}")
        run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)


def ensure_system_command(name: str, apt_packages: Iterable[str]) -> None:
    if shutil.which(name):
        return
    if not shutil.which("apt-get"):
        raise RuntimeError(f"required system command '{name}' is missing and apt-get is unavailable")
    log(f"Installing missing system packages for '{name}': {' '.join(apt_packages)}")
    run(["apt-get", "update"], check=True)
    run(["apt-get", "install", "-y", *apt_packages], check=True)


ensure_python_package("numpy", "numpy")
ensure_python_package("torch", "torch")
ensure_python_package("chess", "python-chess")
ensure_system_command("git", ["git"])
ensure_system_command("g++", ["g++", "build-essential"])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import chess


def detect_runtime() -> dict[str, object]:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        device = xm.xla_device()
        return {"name": "xla", "device": device, "is_xla": True, "xm": xm}
    except Exception:
        pass

    if torch.cuda.is_available():
        return {"name": "cuda", "device": torch.device("cuda"), "is_xla": False, "xm": None}

    return {"name": "cpu", "device": torch.device("cpu"), "is_xla": False, "xm": None}


RUNTIME = detect_runtime()


def authenticated_repo_url() -> str:
    if not GITHUB_PAT.strip():
        return REPO_HTTPS_URL
    user = quote(GITHUB_USERNAME, safe="")
    token = quote(GITHUB_PAT, safe="")
    return f"https://{user}:{token}@github.com/amerameryou1-blip/Mythos.git"


def setup_git(repo_dir: Path) -> None:
    run(["git", "config", "--global", "user.name", GITHUB_USERNAME], check=True)
    run(["git", "config", "--global", "user.email", GITHUB_EMAIL], check=True)
    run(["git", "config", "--global", "pull.rebase", "true"], check=True)
    run(["git", "config", "--global", "--add", "safe.directory", str(repo_dir)], check=True)


def clone_or_update_repo() -> tuple[Path, str]:
    repo_url = authenticated_repo_url()

    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        log(f"Reusing existing repository at {REPO_DIR}")
        run(["git", "remote", "set-url", "origin", repo_url], cwd=REPO_DIR, check=True)
        run(["git", "fetch", "origin"], cwd=REPO_DIR, check=True)
    elif REPO_DIR.exists():
        raise RuntimeError(f"{REPO_DIR} exists but is not a git repository")
    else:
        log(f"Cloning repository into {REPO_DIR}")
        run(["git", "clone", repo_url, str(REPO_DIR)], check=True)

    branch = detect_default_branch(REPO_DIR)
    run(["git", "checkout", branch], cwd=REPO_DIR, check=True)
    run(["git", "pull", "--rebase", "origin", branch], cwd=REPO_DIR, check=False)
    return REPO_DIR, branch


def detect_default_branch(repo_dir: Path) -> str:
    probe = run(
        ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
    )
    if probe.returncode == 0:
        value = probe.stdout.strip()
        if value.startswith("origin/"):
            return value.split("/", 1)[1]
    for candidate in ("main", "master"):
        show = run(["git", "show-ref", "--verify", f"refs/remotes/origin/{candidate}"], cwd=repo_dir, check=False)
        if show.returncode == 0:
            return candidate
    return "main"


def build_engine(repo_dir: Path) -> Path:
    engine_dir = repo_dir / "engine"
    engine_binary = engine_dir / "chess_engine"
    modern_main = engine_dir / "src" / "main.cpp"
    legacy_main = engine_dir / "main.cpp"
    include_dir = engine_dir / "include"

    compile_attempts: list[list[str]] = []

    if modern_main.exists():
        modern_sources = [
            engine_dir / "bitboard.cpp",
            engine_dir / "position.cpp",
            engine_dir / "movegen.cpp",
            engine_dir / "evaluate.cpp",
            engine_dir / "src" / "eval" / "evaluator.cpp",
            engine_dir / "src" / "search" / "search.cpp",
            engine_dir / "src" / "uci" / "uci.cpp",
            engine_dir / "src" / "main.cpp",
        ]
        if all(source.exists() for source in modern_sources):
            compile_attempts.append(
                [
                    "g++",
                    "-O3",
                    "-std=c++20",
                    "-march=native",
                    "-DNDEBUG",
                    "-pthread",
                    "-I",
                    str(include_dir),
                    "-I",
                    str(engine_dir),
                    *[str(source) for source in modern_sources],
                    "-o",
                    str(engine_binary),
                ]
            )

    if legacy_main.exists():
        legacy_sources = sorted(engine_dir.glob("*.cpp"))
        compile_attempts.append(
            [
                "g++",
                "-O3",
                "-std=c++20",
                "-march=native",
                "-DNDEBUG",
                "-pthread",
                *[str(source) for source in legacy_sources],
                "-o",
                str(engine_binary),
            ]
        )

    if (repo_dir / "CMakeLists.txt").exists():
        build_dir = repo_dir / "build"
        compile_attempts.append(["cmake", "-S", str(repo_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
        compile_attempts.append(["cmake", "--build", str(build_dir), "--config", "Release", "-j"])

    last_error = None
    for command in compile_attempts:
        try:
            log(f"Build step: {' '.join(command)}")
            run(command, cwd=repo_dir, check=True)
            if engine_binary.exists():
                log(f"Engine built successfully at {engine_binary}")
                return engine_binary
        except Exception as exc:
            last_error = exc
            log(f"Build attempt failed: {exc}")

    if engine_binary.exists():
        return engine_binary
    raise RuntimeError(f"unable to build engine at {repo_dir}") from last_error


class ClippedReLU(nn.Module):
    def __init__(self, max_val: float = 1.0) -> None:
        super().__init__()
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, self.max_val)


class NNUEModel(nn.Module):
    INPUT_SIZE = 768
    L1_SIZE = 256
    L2_SIZE = 32
    L3_SIZE = 32
    OUTPUT_SIZE = 1
    OUTPUT_SCALE = 400.0

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(self.INPUT_SIZE, self.L1_SIZE)
        self.fc2 = nn.Linear(self.L1_SIZE, self.L2_SIZE)
        self.fc3 = nn.Linear(self.L2_SIZE, self.L3_SIZE)
        self.fc4 = nn.Linear(self.L3_SIZE, self.OUTPUT_SIZE)
        self.activation = ClippedReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x * self.OUTPUT_SCALE

    def save_weights(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        with path.open("wb") as handle:
            handle.write(b"NNUE")
            handle.write(struct.pack("<I", 1))
            handle.write(struct.pack("<I", len(layers)))
            for layer in layers:
                weight = layer.weight.detach().cpu().numpy().astype(np.float32)
                bias = layer.bias.detach().cpu().numpy().astype(np.float32)
                out_size, in_size = weight.shape
                handle.write(struct.pack("<I", in_size))
                handle.write(struct.pack("<I", out_size))
                handle.write(weight.tobytes())
                handle.write(bias.tobytes())

    def load_weights(self, path: Path) -> None:
        with path.open("rb") as handle:
            if handle.read(4) != b"NNUE":
                raise ValueError(f"invalid weights file: {path}")
            version = struct.unpack("<I", handle.read(4))[0]
            if version != 1:
                raise ValueError(f"unsupported weights version: {version}")
            layer_count = struct.unpack("<I", handle.read(4))[0]
            layers = [self.fc1, self.fc2, self.fc3, self.fc4]
            if layer_count != len(layers):
                raise ValueError(f"unexpected layer count: {layer_count}")
            for layer in layers:
                in_size = struct.unpack("<I", handle.read(4))[0]
                out_size = struct.unpack("<I", handle.read(4))[0]
                expected_out, expected_in = layer.weight.shape
                if (in_size, out_size) != (expected_in, expected_out):
                    raise ValueError("weights dimensions do not match the network architecture")
                weight = np.frombuffer(handle.read(in_size * out_size * 4), dtype=np.float32).reshape(out_size, in_size)
                bias = np.frombuffer(handle.read(out_size * 4), dtype=np.float32)
                layer.weight.data.copy_(torch.from_numpy(weight.copy()))
                layer.bias.data.copy_(torch.from_numpy(bias.copy()))


class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer: torch.optim.Optimizer, alpha: float = 0.5, k: int = 5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self._step = 0
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        for group in self.param_groups:
            for param in group["params"]:
                self.state[param] = {"slow_param": param.detach().clone()}

    def zero_grad(self, set_to_none: Optional[bool] = None) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1
        if self._step % self.k == 0:
            for group in self.param_groups:
                for param in group["params"]:
                    slow = self.state[param]["slow_param"]
                    slow.add_(param.data - slow, alpha=self.alpha)
                    param.data.copy_(slow)
        return loss


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cpu") -> None:
        self.decay = decay
        self.device = torch.device(device)
        self.shadow = {
            name: param.detach().to(self.device).clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach().to(self.device), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to_model(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))


@functools.lru_cache(maxsize=200_000)
def fen_to_feature_array(fen: str) -> np.ndarray:
    features = np.zeros(768, dtype=np.float32)
    piece_to_index = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }
    placement = fen.split()[0]
    square = 56
    for char in placement:
        if char == "/":
            square -= 16
        elif char.isdigit():
            square += int(char)
        else:
            plane = piece_to_index[char]
            features[plane * 64 + square] = 1.0
            square += 1
    return features


@dataclass
class EngineReply:
    move: Optional[str]
    nodes: int
    nps: int
    depth: int
    score_cp: int


class UCIEngine:
    def __init__(self, engine_path: Path, hash_mb: int = DEFAULT_HASH_MB, threads: int = TRAIN_THREADS, timeout: float = 60.0) -> None:
        self.engine_path = engine_path
        self.hash_mb = hash_mb
        self.threads = threads
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self.options: set[str] = set()
        self.start()

    def start(self) -> None:
        self.process = subprocess.Popen(
            [str(self.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok", collect_options=True)
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send(f"setoption name Threads value {self.threads}")
        self._send("isready")
        self._read_until("readyok")

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            self._send("quit")
            self.process.wait(timeout=2.0)
        except Exception:
            pass
        finally:
            if self.process.poll() is None:
                self.process.kill()
            self.process = None

    def restart(self) -> None:
        self.stop()
        time.sleep(0.2)
        self.start()

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("engine process is not available")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _readline(self, timeout: float) -> Optional[str]:
        if self.process is None or self.process.stdout is None:
            return None
        try:
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if ready:
                return self.process.stdout.readline().strip()
            return None
        except Exception:
            line = self.process.stdout.readline()
            return line.strip() if line else None

    def _read_until(self, token: str, collect_options: bool = False) -> None:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            line = self._readline(0.5)
            if not line:
                continue
            if collect_options and line.startswith("option name "):
                name = line[len("option name "):].split(" type ", 1)[0].strip()
                if name:
                    self.options.add(name)
            if token in line:
                return
        raise TimeoutError(f"engine timed out waiting for '{token}'")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def set_weights(self, weights_path: Optional[Path]) -> bool:
        if weights_path is None or not weights_path.exists():
            return False
        if "WeightsFile" not in self.options:
            return False
        self._send(f"setoption name WeightsFile value {weights_path}")
        self._send("isready")
        self._read_until("readyok")
        return True

    def best_move(self, fen: str, depth: int) -> EngineReply:
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        deadline = time.time() + self.timeout
        best_move = None
        nodes = 0
        nps = 0
        reached_depth = 0
        score_cp = 0

        while time.time() < deadline:
            line = self._readline(0.5)
            if not line:
                continue
            if line.startswith("info "):
                tokens = line.split()
                for index, token in enumerate(tokens):
                    if token == "depth" and index + 1 < len(tokens):
                        reached_depth = int(tokens[index + 1])
                    elif token == "nodes" and index + 1 < len(tokens):
                        nodes = int(tokens[index + 1])
                    elif token == "nps" and index + 1 < len(tokens):
                        nps = int(tokens[index + 1])
                    elif token == "score" and index + 2 < len(tokens):
                        if tokens[index + 1] == "cp":
                            score_cp = int(tokens[index + 2])
                        elif tokens[index + 1] == "mate":
                            mate = int(tokens[index + 2])
                            score_cp = 30_000 if mate > 0 else -30_000
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] not in ("0000", "(none)"):
                    best_move = parts[1]
                return EngineReply(best_move, nodes, nps, reached_depth, score_cp)

        self.restart()
        return EngineReply(None, nodes, nps, reached_depth, score_cp)


@dataclass
class GameSummary:
    positions: list[tuple[str, int]]
    result: int
    moves: int
    avg_nodes: float
    avg_nps: float
    valid: bool


def play_selfplay_game(engine: UCIEngine, depth: int, max_plies: int = MAX_PLIES) -> GameSummary:
    board = chess.Board()
    positions: list[str] = []
    total_nodes = 0
    total_nps = 0
    engine_moves = 0

    random_opening_plies = random.randint(0, 4)
    for _ in range(random_opening_plies):
        if board.is_game_over(claim_draw=True):
            break
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        board.push(random.choice(legal_moves))

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        positions.append(board.fen())
        reply = engine.best_move(board.fen(), depth=depth)
        if reply.move is None:
            return GameSummary([], 0, board.ply(), 0.0, 0.0, False)
        try:
            move = chess.Move.from_uci(reply.move)
        except ValueError:
            return GameSummary([], 0, board.ply(), 0.0, 0.0, False)
        if move not in board.legal_moves:
            return GameSummary([], 0, board.ply(), 0.0, 0.0, False)
        board.push(move)
        total_nodes += reply.nodes
        total_nps += reply.nps
        engine_moves += 1

    if board.is_checkmate():
        result = 1 if board.outcome(claim_draw=True).winner else -1
    elif board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            result = 0
        else:
            result = 1 if outcome.winner else -1
    else:
        result = 0

    labelled_positions = [(fen, result) for fen in positions]
    avg_nodes = total_nodes / max(1, engine_moves)
    avg_nps = total_nps / max(1, engine_moves)
    return GameSummary(labelled_positions, result, board.ply(), avg_nodes, avg_nps, True)


def write_records(path: Path, records: Iterable[tuple[str, int]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("ab") as handle:
        for fen, result in records:
            fen_bytes = fen.encode("utf-8")
            handle.write(struct.pack("<H", len(fen_bytes)))
            handle.write(fen_bytes)
            handle.write(struct.pack("<b", int(result)))
            written += 1
    return written


def generate_selfplay_shard(
    engine_path: Path,
    weights_path: Optional[Path],
    output_path: Path,
    games: Optional[int],
    time_budget_seconds: Optional[int],
    depth: int,
) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    engine = UCIEngine(engine_path)
    weights_loaded = engine.set_weights(weights_path)
    if weights_path is not None and weights_path.exists() and not weights_loaded:
        log("Engine does not advertise a WeightsFile UCI option; self-play will still run, but new weights will only be exported and logged.")
    engine.new_game()

    started = time.time()
    deadline = started + time_budget_seconds if time_budget_seconds else None
    games_played = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    positions_written = 0
    move_count = 0
    avg_nodes_accum = 0.0
    avg_nps_accum = 0.0

    try:
        while True:
            if games is not None and games_played >= games:
                break
            if deadline is not None and time.time() >= deadline:
                break

            try:
                summary = play_selfplay_game(engine, depth=depth)
            except Exception as exc:
                log(f"Self-play game crashed, restarting engine: {exc}")
                engine.restart()
                engine.set_weights(weights_path)
                engine.new_game()
                continue

            if not summary.valid or not summary.positions:
                engine.restart()
                engine.set_weights(weights_path)
                engine.new_game()
                continue

            positions_written += write_records(output_path, summary.positions)
            games_played += 1
            move_count += summary.moves
            avg_nodes_accum += summary.avg_nodes
            avg_nps_accum += summary.avg_nps

            if summary.result > 0:
                white_wins += 1
            elif summary.result < 0:
                black_wins += 1
            else:
                draws += 1

            if games_played % 5 == 0:
                log(
                    f"Self-play progress: games={games_played} positions={positions_written} "
                    f"white={white_wins} black={black_wins} draws={draws}"
                )

            engine.new_game()
    finally:
        engine.stop()

    return {
        "games": float(games_played),
        "positions": float(positions_written),
        "white_wins": float(white_wins),
        "black_wins": float(black_wins),
        "draws": float(draws),
        "avg_moves_per_game": move_count / max(1, games_played),
        "avg_nodes_per_move": avg_nodes_accum / max(1, games_played),
        "avg_nps": avg_nps_accum / max(1, games_played),
        "weights_loaded": float(1 if weights_loaded else 0),
        "elapsed_seconds": time.time() - started,
    }


def count_records(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        while True:
            raw = handle.read(2)
            if len(raw) < 2:
                break
            fen_length = struct.unpack("<H", raw)[0]
            handle.seek(fen_length + 1, os.SEEK_CUR)
            count += 1
    return count


def recent_training_shards(repo_dir: Path, max_recent: int = MAX_RECENT_SHARDS) -> list[Path]:
    shard_dir = repo_dir / "artifacts" / "selfplay"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(shard_dir.glob("cycle_*.bin"))
    return shards[-max_recent:]


def load_training_tensors(shards: list[Path], max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[np.ndarray] = []
    targets: list[list[float]] = []

    for shard in shards:
        with shard.open("rb") as handle:
            while True:
                raw = handle.read(2)
                if len(raw) < 2:
                    break
                fen_length = struct.unpack("<H", raw)[0]
                fen = handle.read(fen_length).decode("utf-8", errors="ignore")
                result_bytes = handle.read(1)
                if len(result_bytes) < 1:
                    break
                result = struct.unpack("<b", result_bytes)[0]
                features.append(fen_to_feature_array(fen))
                targets.append([float(result) * 400.0])

    if not features:
        raise RuntimeError("no valid training records were loaded")

    if len(features) > max_samples:
        chosen = sorted(random.sample(range(len(features)), max_samples))
        features = [features[index] for index in chosen]
        targets = [targets[index] for index in chosen]

    feature_tensor = torch.from_numpy(np.stack(features).astype(np.float32))
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return feature_tensor, target_tensor


def make_dataloaders(features: torch.Tensor, targets: torch.Tensor, batch_size: int) -> tuple[DataLoader, DataLoader]:
    sample_count = features.shape[0]
    indices = torch.randperm(sample_count)
    val_count = max(1, int(sample_count * 0.1))
    if sample_count <= 4:
        val_count = 1
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]
    if train_indices.numel() == 0:
        train_indices = val_indices

    train_dataset = TensorDataset(features[train_indices], targets[train_indices])
    val_dataset = TensorDataset(features[val_indices], targets[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, is_xla: bool) -> float:
    model.eval()
    total_loss = torch.zeros((), device=device)
    batch_count = 0
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            loss = F.smooth_l1_loss(predictions, batch_targets)
            total_loss = total_loss + loss.detach()
            batch_count += 1
            if is_xla:
                import torch_xla.core.xla_model as xm  # type: ignore

                xm.mark_step()
    if batch_count == 0:
        return 0.0
    return float((total_loss / batch_count).cpu().item())


def train_model(
    model: NNUEModel,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, float]:
    device = RUNTIME["device"]
    is_xla = bool(RUNTIME["is_xla"])
    xm = RUNTIME["xm"]

    train_loader, val_loader = make_dataloaders(features, targets, batch_size=batch_size)
    optimizer = Lookahead(AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2, betas=(0.9, 0.95)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, T_max=max(1, epochs))
    ema = ExponentialMovingAverage(model, decay=0.999, device=str(device))

    model.to(device)
    start_val = evaluate_loss(model, val_loader, device, is_xla)
    last_epoch_loss = start_val

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        batch_count = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = F.smooth_l1_loss(predictions, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if is_xla:
                xm.optimizer_step(optimizer, barrier=False)
                xm.mark_step()
            else:
                optimizer.step()
            ema.update(model)
            epoch_loss = epoch_loss + loss.detach()
            batch_count += 1
        scheduler.step()
        last_epoch_loss = float((epoch_loss / max(1, batch_count)).cpu().item())
        log(f"Training epoch {epoch}/{epochs} loss={last_epoch_loss:.6f}")

    ema.copy_to_model(model)
    end_val = evaluate_loss(model, val_loader, device, is_xla)
    model.to("cpu")

    return {
        "train_loss_last_epoch": last_epoch_loss,
        "val_loss_before": start_val,
        "val_loss_after": end_val,
        "epochs": float(epochs),
        "samples": float(features.shape[0]),
        "batch_size": float(batch_size),
        "learning_rate": float(learning_rate),
    }


def checkpoint_dir(repo_dir: Path) -> Path:
    path = repo_dir / "artifacts" / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_or_create_model(repo_dir: Path) -> NNUEModel:
    model = NNUEModel()
    latest_weights = repo_dir / "weights.bin"
    if latest_weights.exists():
        log(f"Loading existing weights from {latest_weights}")
        model.load_weights(latest_weights)
    return model


def save_model_artifacts(repo_dir: Path, model: NNUEModel, cycle: int, metrics: dict[str, float]) -> tuple[Path, Path]:
    nnue_dir = repo_dir / "artifacts" / "nnue"
    metrics_dir = repo_dir / "artifacts" / "metrics"
    nnue_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    versioned_weights = nnue_dir / f"weights_v{cycle}.bin"
    latest_weights = repo_dir / "weights.bin"
    versioned_metrics = metrics_dir / f"cycle_{cycle:04d}.json"

    model.save_weights(versioned_weights)
    shutil.copy2(versioned_weights, latest_weights)
    torch.save({"cycle": cycle, "metrics": metrics, "state_dict": model.state_dict()}, checkpoint_dir(repo_dir) / "latest.pt")

    with versioned_metrics.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    return versioned_weights, versioned_metrics


def learning_summary(metrics: dict[str, float]) -> str:
    deltas = metrics["val_loss_before"] - metrics["val_loss_after"]
    statements: list[str] = []
    if deltas > 0.01:
        statements.append("Validation loss compressed materially, so the net calibrated fresh self-play positions more cleanly.")
    elif deltas > 0.0:
        statements.append("Validation loss ticked down, which suggests the latest cycle still improved evaluation stability.")
    else:
        statements.append("Validation loss did not improve this hour, so the cycle added exploration more than immediate calibration gains.")

    draw_rate = metrics["draws"] / max(1.0, metrics["games"])
    if draw_rate >= 0.45:
        statements.append("Self-play stayed fairly draw-heavy, which usually means the opening and early middlegame are becoming more stable.")
    else:
        statements.append("Self-play remained decisive enough to keep the result signal sharp.")

    if metrics["white_wins"] > metrics["black_wins"] * 1.15:
        statements.append("White still held a first-move edge in the generated games.")
    elif metrics["black_wins"] > metrics["white_wins"] * 1.15:
        statements.append("Black converted counterplay surprisingly often, which is worth monitoring for bias or search instability.")
    else:
        statements.append("Win rates stayed roughly balanced across colors.")

    if metrics["avg_nodes_per_move"] > 0:
        statements.append(f"Average search effort landed around {metrics['avg_nodes_per_move']:.0f} nodes per move.")
    return " ".join(statements)


def append_training_log(repo_dir: Path, cycle: int, metrics: dict[str, float]) -> None:
    log_path = repo_dir / "TRAINING_LOG.md"
    if not log_path.exists():
        log_path.write_text("# Mythos Training Log\n\n", encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    learned = learning_summary(metrics)
    section = (
        f"## Cycle {cycle} - {timestamp}\n\n"
        f"- Runtime: `{RUNTIME['name']}`\n"
        f"- Games generated: `{int(metrics['games'])}`\n"
        f"- Positions generated: `{int(metrics['positions'])}`\n"
        f"- Results: `W {int(metrics['white_wins'])} / D {int(metrics['draws'])} / B {int(metrics['black_wins'])}`\n"
        f"- Avg moves per game: `{metrics['avg_moves_per_game']:.2f}`\n"
        f"- Avg nodes per move: `{metrics['avg_nodes_per_move']:.2f}`\n"
        f"- Avg NPS: `{metrics['avg_nps']:.2f}`\n"
        f"- Samples trained: `{int(metrics['samples'])}`\n"
        f"- Validation loss: `{metrics['val_loss_before']:.6f} -> {metrics['val_loss_after']:.6f}`\n"
        f"- Last train loss: `{metrics['train_loss_last_epoch']:.6f}`\n"
        f"- Weights file: `artifacts/nnue/weights_v{cycle}.bin`\n"
        f"- Learned this cycle: {learned}\n\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(section)


def git_sync(repo_dir: Path, branch: str, cycle: int) -> None:
    if not GITHUB_PAT.strip():
        log("GITHUB_PAT is empty; skipping git push while keeping local artifacts updated.")
        return

    run(["git", "remote", "set-url", "origin", authenticated_repo_url()], cwd=repo_dir, check=True)
    run(
        [
            "git",
            "add",
            "-f",
            "--",
            "weights.bin",
            "TRAINING_LOG.md",
            "artifacts/nnue",
            "artifacts/metrics",
            "artifacts/checkpoints/latest.pt",
        ],
        cwd=repo_dir,
        check=True,
    )

    staged = run(["git", "diff", "--cached", "--name-only"], cwd=repo_dir, check=True, capture_output=True).stdout.strip()
    if not staged:
        log("No staged changes detected; skipping commit and push.")
        return

    commit_message = f"Auto-Training Cycle {cycle}"
    run(["git", "commit", "-m", commit_message], cwd=repo_dir, check=True)
    try:
        run(["git", "pull", "--rebase", "origin", branch], cwd=repo_dir, check=True)
    except Exception:
        run(["git", "rebase", "--abort"], cwd=repo_dir, check=False)
        raise
    run(["git", "push", "origin", branch], cwd=repo_dir, check=True)


def next_cycle_number(repo_dir: Path) -> int:
    nnue_dir = repo_dir / "artifacts" / "nnue"
    nnue_dir.mkdir(parents=True, exist_ok=True)
    versions = []
    for path in nnue_dir.glob("weights_v*.bin"):
        stem = path.stem.replace("weights_v", "")
        if stem.isdigit():
            versions.append(int(stem))
    return max(versions, default=0) + 1


def bootstrap_if_needed(repo_dir: Path, branch: str, engine_path: Path, model: NNUEModel, cycle: int) -> int:
    latest_weights = repo_dir / "weights.bin"
    if latest_weights.exists():
        return cycle

    log("Starting bootstrap sequence with exactly 10 self-play games")
    shard_path = repo_dir / "artifacts" / "selfplay" / f"cycle_{cycle:04d}.bin"
    shard_metrics = generate_selfplay_shard(
        engine_path=engine_path,
        weights_path=None,
        output_path=shard_path,
        games=BOOTSTRAP_GAMES,
        time_budget_seconds=None,
        depth=BOOTSTRAP_DEPTH,
    )
    features, targets = load_training_tensors([shard_path], max_samples=MAX_TRAINING_SAMPLES)
    train_metrics = train_model(model, features, targets, epochs=12, batch_size=1024, learning_rate=3e-4)
    metrics = {**shard_metrics, **train_metrics, "cycle": float(cycle), "bootstrap": 1.0}
    save_model_artifacts(repo_dir, model, cycle, metrics)
    append_training_log(repo_dir, cycle, metrics)
    try:
        git_sync(repo_dir, branch, cycle)
    except Exception as exc:
        log(f"Bootstrap git sync failed but training will continue: {exc}")
    return cycle + 1


def run_training_cycle(repo_dir: Path, branch: str, engine_path: Path, model: NNUEModel, cycle: int) -> None:
    cycle_start = time.time()
    shard_path = repo_dir / "artifacts" / "selfplay" / f"cycle_{cycle:04d}.bin"
    shard_metrics = generate_selfplay_shard(
        engine_path=engine_path,
        weights_path=repo_dir / "weights.bin",
        output_path=shard_path,
        games=None,
        time_budget_seconds=int(CYCLE_SECONDS * SELFPLAY_BUDGET_RATIO),
        depth=CYCLE_DEPTH,
    )

    shard_count = count_records(shard_path)
    if shard_count == 0:
        raise RuntimeError(f"cycle {cycle} produced no training records")

    shards = recent_training_shards(repo_dir, max_recent=MAX_RECENT_SHARDS)
    features, targets = load_training_tensors(shards, max_samples=MAX_TRAINING_SAMPLES)
    accelerator_batch = 4096 if RUNTIME["name"] in {"cuda", "xla"} else 1024
    train_metrics = train_model(model, features, targets, epochs=4, batch_size=accelerator_batch, learning_rate=2e-4)

    elapsed = time.time() - cycle_start
    metrics = {
        **shard_metrics,
        **train_metrics,
        "cycle": float(cycle),
        "bootstrap": 0.0,
        "training_wall_seconds": elapsed,
    }
    save_model_artifacts(repo_dir, model, cycle, metrics)
    append_training_log(repo_dir, cycle, metrics)

    try:
        git_sync(repo_dir, branch, cycle)
    except Exception as exc:
        log(f"Cycle {cycle} git sync failed; continuing to next hour: {exc}")

    remainder = CYCLE_SECONDS - (time.time() - cycle_start)
    if remainder > 0:
        log(f"Cycle {cycle} completed early; sleeping {remainder:.1f}s before the next hour begins")
        time.sleep(remainder)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log(f"Runtime detected: {RUNTIME['name']} on device {RUNTIME['device']}")
    repo_dir, branch = clone_or_update_repo()
    setup_git(repo_dir)
    engine_path = build_engine(repo_dir)
    model = load_or_create_model(repo_dir)
    cycle = next_cycle_number(repo_dir)
    cycle = bootstrap_if_needed(repo_dir, branch, engine_path, model, cycle)

    while True:
        try:
            log(f"Starting training cycle {cycle}")
            run_training_cycle(repo_dir, branch, engine_path, model, cycle)
        except KeyboardInterrupt:
            log("Training interrupted by user; exiting cleanly.")
            raise
        except Exception as exc:
            error_log = repo_dir / "TRAINING_LOG.md"
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with error_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"## Cycle {cycle} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                    f"- Status: `error`\n"
                    f"- Message: `{type(exc).__name__}: {exc}`\n\n"
                )
            log(f"Cycle {cycle} failed with {type(exc).__name__}: {exc}")
            time.sleep(30)
        finally:
            cycle += 1


if __name__ == "__main__":
    main()
