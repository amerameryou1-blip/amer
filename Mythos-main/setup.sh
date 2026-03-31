#!/bin/bash
set -e

echo "=============================================="
echo "  Chess Bot Setup Script"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root: sudo bash setup.sh"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

echo "Repository directory: $REPO_DIR"
echo ""

# ============================================
# Step 1: Update system and install packages
# ============================================
echo "Step 1: Installing system packages..."

apt-get update -qq

apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    g++ \
    git \
    curl \
    wget \
    > /dev/null 2>&1

print_status "System packages installed"

# ============================================
# Step 2: Install Python packages
# ============================================
echo ""
echo "Step 2: Installing Python packages..."

# Create virtual environment if it doesn't exist
if [ ! -d "$REPO_DIR/venv" ]; then
    python3 -m venv "$REPO_DIR/venv"
    print_status "Virtual environment created"
fi

# Activate virtual environment
source "$REPO_DIR/venv/bin/activate"

# Upgrade pip
pip install --upgrade pip -q

# Install PyTorch CPU-only version (smaller footprint for bot)
pip install torch --index-url https://download.pytorch.org/whl/cpu -q

# Install other dependencies
pip install berserk requests numpy -q

print_status "Python packages installed"

# ============================================
# Step 3: Compile C++ engine
# ============================================
echo ""
echo "Step 3: Compiling C++ chess engine..."

if [ ! -d "$REPO_DIR/engine" ]; then
    print_error "engine/ directory not found!"
    exit 1
fi

# Compile with optimizations
g++ -O3 -std=c++17 -march=native -flto -DNDEBUG \
    "$REPO_DIR/engine/bitboard.cpp" \
    "$REPO_DIR/engine/position.cpp" \
    "$REPO_DIR/engine/movegen.cpp" \
    "$REPO_DIR/engine/evaluate.cpp" \
    "$REPO_DIR/engine/search.cpp" \
    "$REPO_DIR/engine/main.cpp" \
    -o "$REPO_DIR/engine/chess_engine" \
    -lpthread

chmod +x "$REPO_DIR/engine/chess_engine"

print_status "Chess engine compiled: $REPO_DIR/engine/chess_engine"

# Test engine
echo "Testing engine..."
RESPONSE=$(echo -e "uci\nquit" | "$REPO_DIR/engine/chess_engine" 2>/dev/null | head -5)
if echo "$RESPONSE" | grep -q "uciok"; then
    print_status "Engine UCI test passed"
else
    print_warning "Engine may not be responding correctly"
fi

# ============================================
# Step 4: Create systemd service
# ============================================
echo ""
echo "Step 4: Creating systemd service..."

# Get the user who invoked sudo (not root)
REAL_USER="${SUDO_USER:-$USER}"
REAL_GROUP=$(id -gn "$REAL_USER")

cat > /etc/systemd/system/chessbot.service << EOF
[Unit]
Description=Lichess Chess Bot
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_GROUP
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/venv/bin/python3 $REPO_DIR/bot/lichess_bot.py
Restart=always
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=3

# Environment
Environment=PYTHONUNBUFFERED=1
Environment=PATH=$REPO_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin

# Resource limits (for 1GB RAM VM)
MemoryMax=768M
CPUQuota=90%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=chessbot

[Install]
WantedBy=multi-user.target
EOF

print_status "Systemd service file created"

# ============================================
# Step 5: Enable and configure service
# ============================================
echo ""
echo "Step 5: Configuring systemd service..."

# Reload systemd
systemctl daemon-reload

# Enable service to start on boot
systemctl enable chessbot.service > /dev/null 2>&1

print_status "Service enabled (will start on boot)"

# ============================================
# Step 6: Set permissions
# ============================================
echo ""
echo "Step 6: Setting permissions..."

chown -R "$REAL_USER:$REAL_GROUP" "$REPO_DIR"
chmod +x "$REPO_DIR/engine/chess_engine"
chmod 600 "$REPO_DIR/bot/config.py"

print_status "Permissions set"

# ============================================
# Step 7: Create data directory
# ============================================
echo ""
echo "Step 7: Creating data directories..."

mkdir -p "$REPO_DIR/data"
mkdir -p "$REPO_DIR/logs"
chown -R "$REAL_USER:$REAL_GROUP" "$REPO_DIR/data"
chown -R "$REAL_USER:$REAL_GROUP" "$REPO_DIR/logs"

print_status "Data directories created"

# ============================================
# Done!
# ============================================
echo ""
echo "=============================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Edit bot/config.py and add your Lichess API token:"
echo "     nano $REPO_DIR/bot/config.py"
echo ""
echo "  2. Start the bot:"
echo "     sudo systemctl start chessbot"
echo ""
echo "  3. Check bot status:"
echo "     sudo systemctl status chessbot"
echo ""
echo "  4. View live logs:"
echo "     journalctl -u chessbot -f"
echo ""
echo "  5. Stop the bot:"
echo "     sudo systemctl stop chessbot"
echo ""
echo "=============================================="
echo ""

# Check if config has token
if grep -q 'LICHESS_TOKEN = ""' "$REPO_DIR/bot/config.py" 2>/dev/null; then
    print_warning "Don't forget to add your Lichess API token to bot/config.py!"
fi
