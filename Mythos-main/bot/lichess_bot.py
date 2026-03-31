#!/usr/bin/env python3
"""
Lichess Bot - Connects a UCI chess engine to Lichess
Lightweight implementation for 1GB RAM VMs
"""

import subprocess
import threading
import time
import sys
import os
import signal
import logging
from datetime import datetime
from typing import Optional
from queue import Queue, Empty

try:
    import berserk
except ImportError:
    print("Error: berserk not installed. Run: pip install berserk")
    sys.exit(1)

import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class UCIEngine:
    """Manages a UCI engine subprocess."""
    
    def __init__(self, path: str):
        self.path = path
        self.process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        
    def start(self) -> bool:
        """Start the engine process."""
        try:
            self.process = subprocess.Popen(
                [self.path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            
            # Initialize UCI
            self._send("uci")
            if not self._wait_for("uciok", timeout=5.0):
                self.stop()
                return False
            
            # Set options
            if config.ENGINE_HASH_MB > 0:
                self._send(f"setoption name Hash value {config.ENGINE_HASH_MB}")
            
            if config.WEIGHTS_PATH and os.path.exists(config.WEIGHTS_PATH):
                self._send(f"setoption name WeightsFile value {config.WEIGHTS_PATH}")
            
            self._send("isready")
            if not self._wait_for("readyok", timeout=5.0):
                self.stop()
                return False
                
            return True
            
        except FileNotFoundError:
            logger.error(f"Engine not found: {self.path}")
            return False
        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            return False
    
    def stop(self):
        """Stop the engine process."""
        with self.lock:
            if self.process:
                try:
                    self._send("quit")
                    self.process.wait(timeout=2.0)
                except:
                    pass
                try:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
                except:
                    pass
                self.process = None
    
    def _send(self, cmd: str):
        """Send a command to the engine."""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
            except:
                pass
    
    def _readline(self, timeout: float = None) -> Optional[str]:
        """Read a line from engine output."""
        if not self.process or not self.process.stdout:
            return None
        
        if timeout is not None:
            import select
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not ready:
                return None
        
        try:
            line = self.process.stdout.readline()
            return line.strip() if line else None
        except:
            return None
    
    def _wait_for(self, target: str, timeout: float = 10.0) -> bool:
        """Wait for a specific response from engine."""
        start = time.time()
        while time.time() - start < timeout:
            line = self._readline(timeout=0.1)
            if line and target in line:
                return True
        return False
    
    def new_game(self):
        """Start a new game."""
        with self.lock:
            self._send("ucinewgame")
            self._send("isready")
            self._wait_for("readyok", timeout=5.0)
    
    def get_move(self, position: str, wtime: int = 0, btime: int = 0,
                 winc: int = 0, binc: int = 0, is_white: bool = True) -> Optional[str]:
        """Get best move for position."""
        with self.lock:
            if not self.process:
                return None
            
            self._send(position)
            
            # Build go command
            go_cmd = "go"
            
            if config.ENGINE_MOVETIME_MS > 0:
                go_cmd += f" movetime {config.ENGINE_MOVETIME_MS}"
            elif config.ENGINE_DEPTH > 0:
                go_cmd += f" depth {config.ENGINE_DEPTH}"
            elif wtime > 0 or btime > 0:
                go_cmd += f" wtime {wtime} btime {btime}"
                if winc > 0:
                    go_cmd += f" winc {winc}"
                if binc > 0:
                    go_cmd += f" binc {binc}"
            else:
                go_cmd += " depth 6"
            
            self._send(go_cmd)
            
            # Wait for bestmove
            start = time.time()
            while time.time() - start < config.MOVE_TIMEOUT_SEC:
                line = self._readline(timeout=0.5)
                if line and line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        move = parts[1]
                        if move != "(none)":
                            return move
                    return None
            
            logger.warning("Engine timeout waiting for bestmove")
            return None


class GameThread(threading.Thread):
    """Handles a single game in a separate thread."""
    
    def __init__(self, client: berserk.Client, game_id: str, bot_id: str):
        super().__init__(daemon=True)
        self.client = client
        self.game_id = game_id
        self.bot_id = bot_id
        self.engine: Optional[UCIEngine] = None
        self.is_white: bool = True
        self.should_stop = threading.Event()
        
    def run(self):
        """Main game loop."""
        logger.info(f"Game {self.game_id}: Starting")
        
        # Start engine
        self.engine = UCIEngine(config.ENGINE_PATH)
        if not self.engine.start():
            logger.error(f"Game {self.game_id}: Failed to start engine")
            return
        
        self.engine.new_game()
        opponent = "Unknown"
        time_control = "?"
        result = "unknown"
        
        try:
            # Stream game events
            for event in self.client.bots.stream_game_state(self.game_id):
                if self.should_stop.is_set():
                    break
                
                if event['type'] == 'gameFull':
                    # Initial game state
                    self.is_white = event['white'].get('id', '').lower() == self.bot_id.lower()
                    
                    if self.is_white:
                        opponent = event['black'].get('name', event['black'].get('id', 'Anonymous'))
                    else:
                        opponent = event['white'].get('name', event['white'].get('id', 'Anonymous'))
                    
                    clock = event.get('clock')
                    if clock:
                        mins = clock.get('initial', 0) // 60000
                        inc = clock.get('increment', 0) // 1000
                        time_control = f"{mins}+{inc}"
                    
                    logger.info(f"Game {self.game_id}: vs {opponent} ({time_control}), playing {'White' if self.is_white else 'Black'}")
                    
                    state = event.get('state', {})
                    self._handle_state(state, event.get('initialFen', 'startpos'))
                    
                elif event['type'] == 'gameState':
                    self._handle_state(event)
                    
                    status = event.get('status', '')
                    if status in ('mate', 'resign', 'stalemate', 'timeout', 'draw', 
                                  'outoftime', 'cheat', 'noStart', 'aborted', 'variantEnd'):
                        winner = event.get('winner')
                        if winner:
                            if (winner == 'white' and self.is_white) or (winner == 'black' and not self.is_white):
                                result = "win"
                            else:
                                result = "loss"
                        elif status in ('draw', 'stalemate'):
                            result = "draw"
                        else:
                            result = status
                        break
                        
                elif event['type'] == 'chatLine':
                    pass
                    
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Game {self.game_id}: API error: {e}")
            result = "error"
        except Exception as e:
            logger.error(f"Game {self.game_id}: Error: {e}")
            result = "error"
        finally:
            if self.engine:
                self.engine.stop()
            
            log_game_result(opponent, time_control, result, self.game_id)
            logger.info(f"Game {self.game_id}: Ended - {result}")
    
    def _handle_state(self, state: dict, initial_fen: str = 'startpos'):
        """Handle game state update."""
        moves = state.get('moves', '')
        status = state.get('status', 'started')
        
        if status != 'started':
            return
        
        # Determine whose turn it is
        move_list = moves.split() if moves else []
        is_white_turn = len(move_list) % 2 == 0
        
        # Only move on our turn
        if is_white_turn != self.is_white:
            return
        
        # Build position command
        if initial_fen == 'startpos' or not initial_fen:
            position = "position startpos"
        else:
            position = f"position fen {initial_fen}"
        
        if moves:
            position += f" moves {moves}"
        
        # Get time info
        wtime = state.get('wtime', 60000)
        btime = state.get('btime', 60000)
        winc = state.get('winc', 0)
        binc = state.get('binc', 0)
        
        # Handle infinite time
        if isinstance(wtime, float) and wtime == float('inf'):
            wtime = 3600000
        if isinstance(btime, float) and btime == float('inf'):
            btime = 3600000
        
        # Get engine move
        move = self.engine.get_move(position, wtime, btime, winc, binc, self.is_white)
        
        if move:
            try:
                self.client.bots.make_move(self.game_id, move)
                logger.debug(f"Game {self.game_id}: Played {move}")
            except berserk.exceptions.ResponseError as e:
                logger.error(f"Game {self.game_id}: Failed to make move {move}: {e}")
    
    def stop(self):
        """Signal thread to stop."""
        self.should_stop.set()


class LichessBot:
    """Main bot class handling the event stream."""
    
    def __init__(self):
        if not config.LICHESS_TOKEN:
            raise ValueError("LICHESS_TOKEN not set in config.py")
        
        session = berserk.TokenSession(config.LICHESS_TOKEN)
        self.client = berserk.Client(session)
        self.bot_id: str = ""
        self.games: dict[str, GameThread] = {}
        self.games_lock = threading.Lock()
        self.should_stop = threading.Event()
        
    def verify_account(self) -> bool:
        """Verify the bot account."""
        try:
            account = self.client.account.get()
            self.bot_id = account.get('id', '')
            
            if not account.get('title') == 'BOT':
                logger.error(f"Account '{self.bot_id}' is not a bot account!")
                logger.error("Upgrade at: https://lichess.org/account/oauth/token")
                return False
            
            logger.info(f"Connected as bot: {self.bot_id}")
            return True
            
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Failed to verify account: {e}")
            return False
    
    def _should_accept_challenge(self, challenge: dict) -> tuple[bool, str]:
        """Check if challenge should be accepted."""
        if not config.ACCEPT_ALL_CHALLENGES:
            return False, "Not accepting challenges"
        
        # Check concurrent games
        with self.games_lock:
            if len(self.games) >= config.MAX_CONCURRENT_GAMES:
                return False, "Too many concurrent games"
        
        # Check variant
        variant = challenge.get('variant', {}).get('key', 'standard')
        if config.ACCEPTED_VARIANTS and variant not in config.ACCEPTED_VARIANTS:
            return False, f"Variant {variant} not accepted"
        
        # Check time control
        if config.ACCEPTED_TIME_CONTROLS:
            tc = challenge.get('timeControl', {})
            if tc.get('type') == 'clock':
                initial = tc.get('limit', 0) // 60
                inc = tc.get('increment', 0)
                if (initial, inc) not in config.ACCEPTED_TIME_CONTROLS:
                    return False, "Time control not accepted"
        
        return True, ""
    
    def _handle_challenge(self, challenge: dict):
        """Handle incoming challenge."""
        challenge_id = challenge.get('id', '')
        challenger = challenge.get('challenger', {}).get('name', 'Unknown')
        
        accept, reason = self._should_accept_challenge(challenge)
        
        if accept:
            try:
                self.client.bots.accept_challenge(challenge_id)
                logger.info(f"Accepted challenge from {challenger}")
            except berserk.exceptions.ResponseError as e:
                logger.error(f"Failed to accept challenge: {e}")
        else:
            try:
                self.client.bots.decline_challenge(challenge_id, reason="generic")
                logger.info(f"Declined challenge from {challenger}: {reason}")
            except:
                pass
    
    def _handle_game_start(self, game: dict):
        """Handle game start event."""
        game_id = game.get('gameId') or game.get('id', '')
        
        if not game_id:
            return
        
        with self.games_lock:
            if game_id in self.games:
                return
            
            if len(self.games) >= config.MAX_CONCURRENT_GAMES:
                logger.warning(f"Ignoring game {game_id}: at max concurrent games")
                return
            
            thread = GameThread(self.client, game_id, self.bot_id)
            self.games[game_id] = thread
            thread.start()
    
    def _cleanup_finished_games(self):
        """Remove finished game threads."""
        with self.games_lock:
            finished = [gid for gid, t in self.games.items() if not t.is_alive()]
            for gid in finished:
                del self.games[gid]
    
    def run(self):
        """Main event loop."""
        reconnect_attempts = 0
        
        while not self.should_stop.is_set():
            try:
                logger.info("Connecting to event stream...")
                
                for event in self.client.bots.stream_incoming_events():
                    if self.should_stop.is_set():
                        break
                    
                    reconnect_attempts = 0
                    event_type = event.get('type', '')
                    
                    if event_type == 'challenge':
                        self._handle_challenge(event.get('challenge', {}))
                        
                    elif event_type == 'gameStart':
                        self._handle_game_start(event.get('game', {}))
                        
                    elif event_type == 'gameFinish':
                        game_id = event.get('game', {}).get('id', '')
                        with self.games_lock:
                            if game_id in self.games:
                                self.games[game_id].stop()
                    
                    self._cleanup_finished_games()
                
            except berserk.exceptions.ResponseError as e:
                logger.error(f"API error: {e}")
            except Exception as e:
                logger.error(f"Event stream error: {e}")
            
            if self.should_stop.is_set():
                break
            
            reconnect_attempts += 1
            if reconnect_attempts > config.MAX_RECONNECT_ATTEMPTS:
                logger.error("Max reconnect attempts reached, exiting")
                break
            
            logger.info(f"Reconnecting in {config.RECONNECT_DELAY_SEC}s (attempt {reconnect_attempts})...")
            time.sleep(config.RECONNECT_DELAY_SEC)
        
        # Stop all games
        with self.games_lock:
            for thread in self.games.values():
                thread.stop()
            for thread in self.games.values():
                thread.join(timeout=5.0)
    
    def stop(self):
        """Signal bot to stop."""
        self.should_stop.set()


def log_game_result(opponent: str, time_control: str, result: str, game_id: str):
    """Log game result to file."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {game_id} | vs {opponent} | {time_control} | {result}\n"
        
        with open(config.LOG_FILE, 'a') as f:
            f.write(line)
    except Exception as e:
        logger.error(f"Failed to log game result: {e}")


def main():
    """Entry point."""
    logger.info("Lichess Bot starting...")
    
    bot = LichessBot()
    
    if not bot.verify_account():
        sys.exit(1)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received...")
        bot.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
    
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
