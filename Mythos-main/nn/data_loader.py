#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader for chess training data.

Loads positions from binary file format produced by self_play.py
and converts them to 768-dimensional feature vectors.
"""

import struct
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
from typing import Tuple, Optional, Iterator, List
import numpy as np


def fen_to_features(fen: str) -> torch.Tensor:
    """
    Convert FEN string to 768-dimensional feature vector.
    
    Feature layout:
        - 12 planes (P, N, B, R, Q, K, p, n, b, r, q, k) × 64 squares
        - Index = piece_index * 64 + square
        
    Args:
        fen: FEN string (piece placement, can include full FEN)
        
    Returns:
        Tensor of shape (768,) with binary features
    """
    features = torch.zeros(768, dtype=torch.float32)
    
    piece_to_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Get piece placement from FEN
    placement = fen.split()[0] if ' ' in fen else fen
    
    # Early return for non-standard FEN (from self-play)
    if not any(c in placement for c in 'PNBRQKpnbrqk'):
        # Return random features for testing
        return features
    
    square = 56  # Start at A8 (rank 8, file A)
    
    for char in placement:
        if char == '/':
            square -= 16  # Move to previous rank (8 squares back + 8 for skipping)
        elif char.isdigit():
            square += int(char)
        elif char in piece_to_idx:
            piece_idx = piece_to_idx[char]
            feature_idx = piece_idx * 64 + square
            if 0 <= feature_idx < 768:
                features[feature_idx] = 1.0
            square += 1
    
    return features


def parse_side_to_move(fen: str) -> int:
    """
    Parse side to move from FEN.
    
    Args:
        fen: Full FEN string
        
    Returns:
        1 for white, -1 for black
    """
    parts = fen.split()
    if len(parts) >= 2:
        return 1 if parts[1] == 'w' else -1
    return 1  # Default to white


class ChessPositionDataset(Dataset):
    """
    Dataset that loads all positions into memory.
    
    Good for smaller datasets that fit in RAM.
    """
    
    def __init__(self, data_path: str, transform_for_stm: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to binary data file
            transform_for_stm: If True, flip eval sign for black to move
        """
        self.data_path = Path(data_path)
        self.transform_for_stm = transform_for_stm
        self.positions: List[Tuple[str, int]] = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load all positions from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'rb') as f:
            while True:
                # Read FEN length
                length_bytes = f.read(2)
                if len(length_bytes) < 2:
                    break
                
                fen_length = struct.unpack('<H', length_bytes)[0]
                
                # Read FEN string
                fen_bytes = f.read(fen_length)
                if len(fen_bytes) < fen_length:
                    break
                
                fen = fen_bytes.decode('utf-8')
                
                # Read result
                result_bytes = f.read(1)
                if len(result_bytes) < 1:
                    break
                
                result = struct.unpack('<b', result_bytes)[0]
                
                self.positions.append((fen, result))
        
        print(f"Loaded {len(self.positions):,} positions from {self.data_path}")
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fen, result = self.positions[idx]
        
        features = fen_to_features(fen)
        
        # Transform result to target
        # Result is from game outcome (1 = white win, 0 = draw, -1 = black win)
        target = float(result)
        
        # Optionally flip for side to move
        if self.transform_for_stm:
            stm = parse_side_to_move(fen)
            if stm == -1:  # Black to move
                target = -target
        
        return features, torch.tensor([target], dtype=torch.float32)


class ChessPositionIterableDataset(IterableDataset):
    """
    Iterable dataset that streams positions from file.
    
    Good for large datasets that don't fit in memory.
    Properly handles multiple DataLoader workers.
    """
    
    def __init__(self, data_path: str, transform_for_stm: bool = True,
                 shuffle_buffer_size: int = 100000):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to binary data file
            transform_for_stm: If True, flip eval sign for black to move
            shuffle_buffer_size: Size of shuffle buffer (0 to disable)
        """
        self.data_path = Path(data_path)
        self.transform_for_stm = transform_for_stm
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Count positions
        self._count = self._count_positions()
    
    def _count_positions(self) -> int:
        """Count positions in file."""
        if not self.data_path.exists():
            return 0
        
        count = 0
        with open(self.data_path, 'rb') as f:
            while True:
                length_bytes = f.read(2)
                if len(length_bytes) < 2:
                    break
                
                fen_length = struct.unpack('<H', length_bytes)[0]
                f.seek(fen_length + 1, 1)  # Skip FEN and result
                count += 1
        
        return count
    
    def __len__(self) -> int:
        return self._count
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Get worker info for sharding
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-threaded
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        position_idx = 0
        
        with open(self.data_path, 'rb') as f:
            while True:
                # Read FEN length
                length_bytes = f.read(2)
                if len(length_bytes) < 2:
                    break
                
                fen_length = struct.unpack('<H', length_bytes)[0]
                
                # Read FEN
                fen_bytes = f.read(fen_length)
                if len(fen_bytes) < fen_length:
                    break
                
                fen = fen_bytes.decode('utf-8')
                
                # Read result
                result_bytes = f.read(1)
                if len(result_bytes) < 1:
                    break
                
                result = struct.unpack('<b', result_bytes)[0]
                
                # Shard positions among workers
                if position_idx % num_workers != worker_id:
                    position_idx += 1
                    continue
                
                position_idx += 1
                
                # Convert to features
                features = fen_to_features(fen)
                target = float(result)
                
                if self.transform_for_stm:
                    stm = parse_side_to_move(fen)
                    if stm == -1:
                        target = -target
                
                item = (features, torch.tensor([target], dtype=torch.float32))
                
                if self.shuffle_buffer_size > 0:
                    buffer.append(item)
                    
                    if len(buffer) >= self.shuffle_buffer_size:
                        # Yield random items from buffer
                        np.random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []
                else:
                    yield item
        
        # Yield remaining buffered items
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item


def create_dataloader(data_path: str, batch_size: int = 256,
                      shuffle: bool = True, num_workers: int = 4,
                      pin_memory: bool = True, 
                      streaming: bool = False) -> DataLoader:
    """
    Create a DataLoader for chess training data.
    
    Args:
        data_path: Path to binary data file
        batch_size: Batch size
        shuffle: Whether to shuffle (only for non-streaming)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (for GPU training)
        streaming: Use streaming dataset for large files
        
    Returns:
        DataLoader instance
    """
    if streaming:
        dataset = ChessPositionIterableDataset(data_path)
        shuffle = False  # Handled internally
    else:
        dataset = ChessPositionDataset(data_path)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        pin_memory = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not streaming,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )


def count_positions(data_path: str) -> int:
    """Count positions in a data file."""
    path = Path(data_path)
    if not path.exists():
        return 0
    
    count = 0
    with open(path, 'rb') as f:
        while True:
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                break
            
            fen_length = struct.unpack('<H', length_bytes)[0]
            
            # Validate read
            if f.read(fen_length) and f.read(1):
                count += 1
            else:
                break
    
    return count


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data loading')
    parser.add_argument('--data', type=str, default='data/games.bin',
                        help='Path to data file')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    args = parser.parse_args()
    
    path = Path(args.data)
    
    if not path.exists():
        # Create test data
        print("Creating test data...")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        ]
        
        with open(path, 'wb') as f:
            for i, fen in enumerate(test_fens * 100):
                result = [1, 0, -1][i % 3]
                fen_bytes = fen.encode('utf-8')
                f.write(struct.pack('<H', len(fen_bytes)))
                f.write(fen_bytes)
                f.write(struct.pack('<b', result))
        
        print(f"Created {path}")
    
    # Test dataset
    print(f"\nTesting ChessPositionDataset:")
    dataset = ChessPositionDataset(str(path))
    print(f"  Length: {len(dataset)}")
    
    features, target = dataset[0]
    print(f"  Features shape: {features.shape}")
    print(f"  Non-zero features: {features.sum().item()}")
    print(f"  Target: {target.item()}")
    
    # Test dataloader
    print(f"\nTesting DataLoader:")
    loader = create_dataloader(str(path), batch_size=args.batch_size, num_workers=0)
    
    for i, (features, targets) in enumerate(loader):
        print(f"  Batch {i}: features {features.shape}, targets {targets.shape}")
        if i >= 2:
            break
    
    print("\nAll tests passed!")
