"""
Lichess Bot Configuration
Fill in your Lichess bot token below.
"""

# Lichess API token (get from https://lichess.org/account/oauth/token)
# Required scopes: bot:play, challenge:read, challenge:write
LICHESS_TOKEN = ""

# Path to compiled UCI engine executable
ENGINE_PATH = "./engine/chess_engine"

# Path to NNUE weights file (optional, engine loads if exists)
WEIGHTS_PATH = "./weights.bin"

# Auto-accept all incoming challenges
ACCEPT_ALL_CHALLENGES = True

# Accept only these time controls (empty = accept all)
# Format: (initial_minutes, increment_seconds)
ACCEPTED_TIME_CONTROLS: list[tuple[int, int]] = []

# Accept only these variants (empty = accept all)
# Options: "standard", "chess960", etc.
ACCEPTED_VARIANTS: list[str] = ["standard"]

# Maximum number of concurrent games
MAX_CONCURRENT_GAMES = 4

# Log file for game results
LOG_FILE = "games_log.txt"

# Engine settings
ENGINE_DEPTH = 0  # 0 = use time management
ENGINE_MOVETIME_MS = 0  # 0 = use time management
ENGINE_HASH_MB = 64  # Transposition table size

# Reconnect settings
RECONNECT_DELAY_SEC = 5
MAX_RECONNECT_ATTEMPTS = 10

# Move timeout (kill engine if no response)
MOVE_TIMEOUT_SEC = 30.0
