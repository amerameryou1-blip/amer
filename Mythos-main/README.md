# Chess Bot — Train on Kaggle, Play on Lichess

A complete chess engine with NNUE-style neural network evaluation, trained via self-play and deployed as a Lichess bot.

## Architecture

```
├── engine/          # C++ UCI chess engine
│   ├── types.h      # Core types, enums, Move struct
│   ├── bitboard.*   # Magic bitboards, attack tables
│   ├── position.*   # Board state, FEN, make/unmake move
│   ├── movegen.*    # Legal move generation
│   ├── evaluate.*   # Piece-square tables, tapered eval
│   ├── search.*     # Alpha-beta, LMR, null move, TT
│   └── main.cpp     # UCI protocol handler
├── nn/              # Python training code
│   ├── nnue_model.py    # PyTorch NNUE network (768→256→32→32→1)
│   ├── self_play.py     # Generate training data via engine self-play
│   ├── data_loader.py   # PyTorch Dataset for training
│   └── convert_weights.py  # Export weights to C++ format
├── bot/             # Lichess bot
│   ├── config.py    # API token and settings
│   └── lichess_bot.py   # Berserk-based bot runner
├── setup.sh         # One-command server setup
└── README.md        # This file
```

---

## Section 1 — Training (on Kaggle)

Train the neural network using Kaggle's free GPU.

### Step 1: Create Kaggle Notebook

1. Go to [kaggle.com](https://kaggle.com) and sign in
2. Click **"Create"** → **"New Notebook"**
3. In the right panel, under **Accelerator**, select **GPU P100**
4. Rename notebook to "Chess NNUE Training"

### Step 2: Clone Repository

In the first cell, run:

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

### Step 3: Generate Training Data

```python
# Install dependencies
!pip install torch numpy

# Compile the engine
!g++ -O3 -std=c++17 -march=native engine/*.cpp -o engine/chess_engine -lpthread

# Generate self-play games (adjust count as needed)
!python nn/self_play.py --games 10000 --depth 6 --workers 2
```

This creates `data/games.bin` with positions and outcomes.

### Step 4: Train the Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
from nn.nnue_model import NNUEModel
from nn.data_loader import ChessDataset, get_dataloader

# Load data
dataset = ChessDataset('data/games.bin')
train_loader = get_dataloader(dataset, batch_size=4096, shuffle=True)

# Create model
model = NNUEModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(100):
    total_loss = 0
    for features, targets in train_loader:
        features = features.cuda()
        targets = targets.cuda()
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

# Save checkpoint
torch.save(model.state_dict(), 'model_checkpoint.pt')
print("Training complete!")
```

### Step 5: Export Weights

```python
from nn.convert_weights import export_weights

# Convert to C++ format
export_weights('model_checkpoint.pt', 'weights.bin')
print("Weights exported to weights.bin")
```

### Step 6: Download

```python
# Create zip with everything needed
!zip -r chess_bot.zip weights.bin engine/ bot/

# Download link will appear
from IPython.display import FileLink
FileLink('chess_bot.zip')
```

---

## Section 2 — Deploying the Bot

Deploy on a free Oracle Cloud VM.

### Step 1: Create Oracle Cloud VM

1. Go to [cloud.oracle.com](https://cloud.oracle.com) and create a free account
2. Navigate to **Compute** → **Instances** → **Create Instance**
3. Configure:
   - **Shape**: VM.Standard.E2.1.Micro (Always Free)
   - **Image**: Ubuntu 22.04
   - **Boot volume**: 50 GB
4. Download SSH key and click **Create**
5. Note the **Public IP address**

### Step 2: Connect to VM

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP
```

### Step 3: Upload Repository

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# If you trained locally, upload weights
scp -i your-key.pem weights.bin ubuntu@YOUR_PUBLIC_IP:~/YOUR_REPO/
```

### Step 4: Run Setup Script

```bash
sudo bash setup.sh
```

This will:
- Install Python 3, pip, g++, git
- Install berserk, requests, torch (CPU)
- Compile the C++ engine
- Create a systemd service

### Step 5: Configure Lichess Token

1. Go to [lichess.org/account/oauth/token](https://lichess.org/account/oauth/token)
2. Create a token with scopes:
   - `bot:play`
   - `challenge:read`
   - `challenge:write`
3. Edit config:

```bash
nano bot/config.py
```

4. Paste your token:

```python
LICHESS_TOKEN = "lip_xxxxxxxxxxxxxxxxxxxxxxxx"
```

5. Save and exit (Ctrl+X, Y, Enter)

### Step 6: Upgrade to Bot Account

**⚠️ Warning: This is irreversible! Your account becomes bot-only.**

1. Go to [lichess.org/api#tag/Bot/operation/botAccountUpgrade](https://lichess.org/api#tag/Bot/operation/botAccountUpgrade)
2. Or run:

```bash
curl -X POST https://lichess.org/api/bot/account/upgrade \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Step 7: Start the Bot

```bash
# Start the service
sudo systemctl start chessbot

# Check status
sudo systemctl status chessbot

# View live logs
journalctl -u chessbot -f
```

### Useful Commands

```bash
# Stop bot
sudo systemctl stop chessbot

# Restart bot
sudo systemctl restart chessbot

# View recent logs
journalctl -u chessbot --since "10 minutes ago"

# Check if engine works
echo -e "uci\nquit" | ./engine/chess_engine
```

---

## Section 3 — Watching Games

### Live Games

1. Go to `https://lichess.org/@/YOUR_BOT_NAME`
2. All games are listed publicly on the profile
3. Click any game to watch the replay move-by-move
4. Use arrow keys to navigate through moves

### Following Your Bot

1. Click **"Follow"** on your bot's profile
2. You'll get notifications when it plays
3. Watch games live as they happen

### Rating Progress

- Bot rating updates after each game
- Starts at 1500 (or adjusted after first games)
- View rating graph on profile page
- Different ratings for each time control:
  - Bullet (< 3 min)
  - Blitz (3-8 min)
  - Rapid (8-25 min)
  - Classical (> 25 min)

### Game Analysis

After each game:
1. Click the game on your bot's profile
2. Click **"Request a computer analysis"**
3. Lichess shows accuracy %, blunders, best moves
4. Compare your engine's moves to Stockfish

### Exporting Games

Download all bot games:

```bash
curl https://lichess.org/api/games/user/YOUR_BOT_NAME \
  -H "Accept: application/x-chess-pgn" \
  > all_games.pgn
```

---

## Configuration Options

Edit `bot/config.py` to customize:

```python
# Lichess API token (required)
LICHESS_TOKEN = "lip_xxxxx"

# Engine settings
ENGINE_PATH = "./engine/chess_engine"
WEIGHTS_PATH = "./weights.bin"

# Game settings
MAX_CONCURRENT_GAMES = 4      # Parallel games
ACCEPT_ALL_CHALLENGES = True  # Auto-accept challenges

# Time controls to accept
ACCEPT_BULLET = True    # < 3 min
ACCEPT_BLITZ = True     # 3-8 min
ACCEPT_RAPID = True     # 8-25 min
ACCEPT_CLASSICAL = False # > 25 min (long games)

# Variants to accept
ACCEPT_STANDARD = True
ACCEPT_CHESS960 = False
```

---

## Troubleshooting

### Bot not starting

```bash
# Check logs for errors
journalctl -u chessbot -n 50

# Common issues:
# - Invalid token: regenerate at lichess.org
# - Not a bot account: upgrade account first
# - Engine not compiled: run setup.sh again
```

### Engine crashes

```bash
# Test engine manually
cd ~/YOUR_REPO
./engine/chess_engine

# Type these commands:
uci
position startpos
go depth 10
quit
```

### Out of memory (1GB VM)

```bash
# Check memory usage
free -h

# Reduce concurrent games in config.py
MAX_CONCURRENT_GAMES = 2

# Restart
sudo systemctl restart chessbot
```

### Rate limited by Lichess

- Lichess limits: 1 request/second sustained
- Bot auto-handles this with backoff
- If persistent, wait 1 hour

---

## License

MIT License. Use freely for learning and fun.

---

## Credits

- [Lichess](https://lichess.org) — Free chess platform
- [Berserk](https://github.com/lichess-org/berserk) — Python Lichess API client
- [Kaggle](https://kaggle.com) — Free GPU for training
- [Oracle Cloud](https://cloud.oracle.com) — Free VM for hosting
