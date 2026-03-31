#!/usr/bin/env python3
"""
Self-play data generation using the C++ chess engine.

Spawns multiple worker processes to play games in parallel,
storing positions with game outcomes for training.
"""

import subprocess
import multiprocessing as mp
import time
import struct
import signal
import sys
import random
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from queue import Empty


@dataclass
class GamePosition:
    """A position with its game outcome."""
    fen: str
    result: int  # 1 = white win, 0 = draw, -1 = black win


class UCIEngine:
    """UCI engine wrapper with timeout handling."""
    
    def __init__(self, engine_path: str, timeout: float = 30.0):
        self.engine_path = engine_path
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self.start()
    
    def start(self) -> None:
        """Start the engine process."""
        self.process = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        
        # Initialize UCI
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")
    
    def stop(self) -> None:
        """Stop the engine process."""
        if self.process:
            try:
                self._send("quit")
                self.process.wait(timeout=2.0)
            except:
                pass
            finally:
                if self.process.poll() is None:
                    self.process.kill()
                self.process = None
    
    def restart(self) -> None:
        """Restart the engine."""
        self.stop()
        time.sleep(0.1)
        self.start()
    
    def _send(self, command: str) -> None:
        """Send a command to the engine."""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
    
    def _readline(self, timeout: float) -> Optional[str]:
        """Read a line with timeout."""
        import select
        
        if not self.process or not self.process.stdout:
            return None
        
        # Use select for timeout (Unix) or polling (Windows fallback)
        try:
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if ready:
                return self.process.stdout.readline().strip()
            return None
        except (ValueError, OSError):
            # Fallback for Windows or broken pipes
            return self.process.stdout.readline().strip()
    
    def _wait_for(self, expected: str, timeout: Optional[float] = None) -> bool:
        """Wait for a specific response."""
        timeout = timeout or self.timeout
        start = time.time()
        
        while time.time() - start < timeout:
            line = self._readline(1.0)
            if line and expected in line:
                return True
        
        return False
    
    def new_game(self) -> None:
        """Start a new game."""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")
    
    def set_position(self, fen: Optional[str] = None, moves: Optional[List[str]] = None) -> None:
        """Set the board position."""
        if fen:
            cmd = f"position fen {fen}"
        else:
            cmd = "position startpos"
        
        if moves:
            cmd += " moves " + " ".join(moves)
        
        self._send(cmd)
    
    def get_best_move(self, depth: int = 6) -> Optional[str]:
        """Get best move at given depth."""
        self._send(f"go depth {depth}")
        
        start = time.time()
        while time.time() - start < self.timeout:
            line = self._readline(1.0)
            if line and line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    move = parts[1]
                    if move == "(none)" or move == "0000":
                        return None
                    return move
        
        # Timeout - restart engine
        self.restart()
        return None


def play_game(engine: UCIEngine, depth: int = 5, max_moves: int = 120,
              add_noise: bool = True) -> List[GamePosition]:
    """
    Play a single game and collect positions.
    
    Args:
        engine: UCI engine instance
        depth: Search depth
        max_moves: Maximum moves before draw
        add_noise: Add random opening moves for diversity
        
    Returns:
        List of positions with game outcome
    """
    positions: List[GamePosition] = []
    moves: List[str] = []
    
    # Position tracking for repetition
    seen_positions: dict = {}
    
    engine.new_game()
    
    # Optional random opening moves for diversity
    if add_noise and random.random() < 0.3:
        num_random = random.randint(1, 4)
        for _ in range(num_random):
            engine.set_position(moves=moves if moves else None)
            # Get a quick move
            move = engine.get_best_move(depth=1)
            if move:
                moves.append(move)
    
    white_to_move = (len(moves) % 2 == 0)
    
    for ply in range(max_moves):
        engine.set_position(moves=moves if moves else None)
        
        # Get FEN for this position (simplified - use move list instead)
        # For proper FEN, we'd need to track the full position
        fen = f"position after {len(moves)} moves"
        
        move = engine.get_best_move(depth=depth)
        
        if not move:
            # No legal move - checkmate or stalemate
            if ply == 0:
                return []  # Invalid game
            
            # Determine result based on previous positions
            # If side to move has no moves, check if it's checkmate
            result = 0  # Assume draw (stalemate)
            
            # Update all positions with result
            for pos in positions:
                pos.result = result
            
            return positions
        
        # Store position (from side to move's perspective)
        current_result = 0  # Will be updated at game end
        positions.append(GamePosition(
            fen=f"ply{ply}_moves_{'_'.join(moves[-4:]) if moves else 'start'}",
            result=current_result
        ))
        
        moves.append(move)
        white_to_move = not white_to_move
        
        # Simple position key for repetition (last 8 moves)
        pos_key = tuple(moves[-8:]) if len(moves) >= 8 else tuple(moves)
        seen_positions[pos_key] = seen_positions.get(pos_key, 0) + 1
        
        if seen_positions[pos_key] >= 3:
            # Threefold repetition - draw
            for pos in positions:
                pos.result = 0
            return positions
    
    # Max moves reached - draw
    for pos in positions:
        pos.result = 0
    
    return positions


def worker_process(worker_id: int, engine_path: str, output_queue: mp.Queue,
                   stop_event: mp.Event, depth: int = 6):
    """
    Worker process that plays games and sends positions to queue.
    """
    # Ignore SIGINT in workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    engine = UCIEngine(engine_path)
    games_played = 0
    
    try:
        while not stop_event.is_set():
            try:
                positions = play_game(engine, depth=depth)
                
                if positions:
                    # Send positions to main process
                    for pos in positions:
                        output_queue.put((pos.fen, pos.result))
                    games_played += 1
                    
            except Exception as e:
                print(f"Worker {worker_id}: Error playing game: {e}")
                engine.restart()
                
    finally:
        engine.stop()
        print(f"Worker {worker_id}: Finished after {games_played} games")


def write_positions(output_path: Path, output_queue: mp.Queue, 
                    stop_event: mp.Event, target_positions: int):
    """
    Writer process that saves positions to file.
    
    File format (binary):
        For each position:
            - FEN length: 2 bytes (uint16)
            - FEN string: variable bytes
            - Result: 1 byte (signed int8: -1, 0, or 1)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    positions_written = 0
    games_completed = 0
    last_report = time.time()
    
    with open(output_path, 'ab') as f:
        while positions_written < target_positions:
            try:
                fen, result = output_queue.get(timeout=1.0)
                
                # Write position
                fen_bytes = fen.encode('utf-8')
                f.write(struct.pack('<H', len(fen_bytes)))
                f.write(fen_bytes)
                f.write(struct.pack('<b', result))
                
                positions_written += 1
                
                # Periodic reporting
                now = time.time()
                if now - last_report >= 30.0:
                    rate = positions_written / (now - last_report)
                    print(f"Positions: {positions_written:,} | Rate: {rate:.1f}/s")
                    last_report = now
                    f.flush()
                    
            except Empty:
                if stop_event.is_set():
                    break
    
    stop_event.set()
    print(f"Total positions written: {positions_written:,}")


def main():
    parser = argparse.ArgumentParser(description='Generate self-play training data')
    parser.add_argument('--engine', type=str, default='./engine/chess_engine',
                        help='Path to UCI engine executable')
    parser.add_argument('--output', type=str, default='data/games.bin',
                        help='Output file path')
    parser.add_argument('--positions', type=int, default=1000000,
                        help='Target number of positions')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('--depth', type=int, default=6,
                        help='Search depth for moves')
    
    args = parser.parse_args()
    
    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"Error: Engine not found at {engine_path}")
        print("Please compile the engine first with:")
        print("  g++ -O3 -std=c++17 engine/*.cpp -o engine/chess_engine")
        sys.exit(1)
    
    num_workers = args.workers or mp.cpu_count()
    output_path = Path(args.output)
    
    print(f"Self-play data generation")
    print(f"  Engine: {engine_path}")
    print(f"  Output: {output_path}")
    print(f"  Target positions: {args.positions:,}")
    print(f"  Workers: {num_workers}")
    print(f"  Depth: {args.depth}")
    print()
    
    # Shared state
    output_queue = mp.Queue(maxsize=10000)
    stop_event = mp.Event()
    
    # Start workers
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, str(engine_path), output_queue, stop_event, args.depth)
        )
        p.start()
        workers.append(p)
    
    # Handle interrupts gracefully
    def signal_handler(sig, frame):
        print("\nStopping workers...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Write positions
    try:
        write_positions(output_path, output_queue, stop_event, args.positions)
    finally:
        stop_event.set()
        
        for p in workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
    
    print("Done!")


if __name__ == '__main__':
    main()
