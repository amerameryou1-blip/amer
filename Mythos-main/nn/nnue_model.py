#!/usr/bin/env python3
"""
NNUE-style neural network for chess position evaluation.

Architecture: 768 -> 256 -> 32 -> 32 -> 1
Input: 768 features (12 piece types × 64 squares)
Output: Centipawn evaluation from white's perspective
"""

import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple


class ClippedReLU(nn.Module):
    """ReLU clamped to [0, 1] for quantization-friendly activations."""
    
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, self.max_val)


class NNUEModel(nn.Module):
    """
    NNUE-style evaluation network.
    
    Input features (768 total):
        - 6 piece types × 2 colors × 64 squares = 768
        - One-hot encoding: 1 if piece present, 0 otherwise
    
    Piece order: P, N, B, R, Q, K for white, then p, n, b, r, q, k for black
    Square order: A1, B1, ..., H1, A2, ..., H8 (0-63)
    
    Feature index = piece_index * 64 + square
    where piece_index = (color * 6) + piece_type
    """
    
    INPUT_SIZE = 768
    L1_SIZE = 256
    L2_SIZE = 32
    L3_SIZE = 32
    OUTPUT_SIZE = 1
    
    # Scale factor for output (centipawns)
    OUTPUT_SCALE = 400.0
    
    def __init__(self):
        super().__init__()
        
        # Network layers
        self.fc1 = nn.Linear(self.INPUT_SIZE, self.L1_SIZE)
        self.fc2 = nn.Linear(self.L1_SIZE, self.L2_SIZE)
        self.fc3 = nn.Linear(self.L2_SIZE, self.L3_SIZE)
        self.fc4 = nn.Linear(self.L3_SIZE, self.OUTPUT_SIZE)
        
        # Activation
        self.activation = ClippedReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 768) with binary features
            
        Returns:
            Evaluation in centipawns, shape (batch, 1)
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # No activation on output
        
        # Scale to centipawn range
        return x * self.OUTPUT_SCALE
    
    def evaluate_position(self, features: torch.Tensor) -> float:
        """
        Evaluate a single position.
        
        Args:
            features: Tensor of shape (768,) with binary features
            
        Returns:
            Evaluation in centipawns from white's perspective
        """
        self.eval()
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)
            output = self.forward(features)
            return output.item()
    
    def save_weights(self, path: str) -> None:
        """
        Save weights to binary format readable by C++.
        
        Format:
            - Magic number: 4 bytes ('NNUE')
            - Version: 4 bytes (uint32)
            - Layer count: 4 bytes (uint32)
            - For each layer:
                - Input size: 4 bytes (uint32)
                - Output size: 4 bytes (uint32)
                - Weights: input_size * output_size * 4 bytes (float32, row-major)
                - Biases: output_size * 4 bytes (float32)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        
        with open(path, 'wb') as f:
            # Header
            f.write(b'NNUE')  # Magic
            f.write(struct.pack('<I', 1))  # Version
            f.write(struct.pack('<I', len(layers)))  # Layer count
            
            for layer in layers:
                weight = layer.weight.detach().cpu().numpy().astype(np.float32)
                bias = layer.bias.detach().cpu().numpy().astype(np.float32)
                
                out_size, in_size = weight.shape
                
                f.write(struct.pack('<I', in_size))
                f.write(struct.pack('<I', out_size))
                f.write(weight.tobytes())
                f.write(bias.tobytes())
        
        print(f"Saved weights to {path}")
        print(f"  Total size: {path.stat().st_size} bytes")
    
    def load_weights(self, path: str) -> None:
        """
        Load weights from binary format.
        
        Args:
            path: Path to weights file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        
        with open(path, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'NNUE':
                raise ValueError(f"Invalid magic number: {magic}")
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != 1:
                raise ValueError(f"Unsupported version: {version}")
            
            layer_count = struct.unpack('<I', f.read(4))[0]
            if layer_count != len(layers):
                raise ValueError(f"Layer count mismatch: {layer_count} != {len(layers)}")
            
            for layer in layers:
                in_size = struct.unpack('<I', f.read(4))[0]
                out_size = struct.unpack('<I', f.read(4))[0]
                
                expected_out, expected_in = layer.weight.shape
                if in_size != expected_in or out_size != expected_out:
                    raise ValueError(f"Layer size mismatch: ({in_size}, {out_size}) != ({expected_in}, {expected_out})")
                
                weight_bytes = f.read(in_size * out_size * 4)
                bias_bytes = f.read(out_size * 4)
                
                weight = np.frombuffer(weight_bytes, dtype=np.float32).reshape(out_size, in_size)
                bias = np.frombuffer(bias_bytes, dtype=np.float32)
                
                layer.weight.data = torch.from_numpy(weight.copy())
                layer.bias.data = torch.from_numpy(bias.copy())
        
        print(f"Loaded weights from {path}")


def fen_to_features(fen: str) -> torch.Tensor:
    """
    Convert FEN string to 768-dimensional feature vector.
    
    Args:
        fen: FEN string (only piece placement part is used)
        
    Returns:
        Tensor of shape (768,) with binary features
    """
    features = torch.zeros(768, dtype=torch.float32)
    
    piece_to_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Get piece placement from FEN
    placement = fen.split()[0]
    
    square = 56  # Start at A8
    for char in placement:
        if char == '/':
            square -= 16  # Move to next rank
        elif char.isdigit():
            square += int(char)
        elif char in piece_to_idx:
            piece_idx = piece_to_idx[char]
            feature_idx = piece_idx * 64 + square
            features[feature_idx] = 1.0
            square += 1
    
    return features


def create_model() -> NNUEModel:
    """Create a new NNUE model."""
    return NNUEModel()


def load_model(path: str) -> NNUEModel:
    """Load a model from weights file."""
    model = NNUEModel()
    model.load_weights(path)
    return model


if __name__ == '__main__':
    # Test the model
    model = NNUEModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    features = fen_to_features(start_fen)
    
    print(f"Features shape: {features.shape}")
    print(f"Non-zero features: {features.sum().item()}")
    
    eval_score = model.evaluate_position(features)
    print(f"Evaluation: {eval_score:.2f} cp")
    
    # Test save/load
    model.save_weights("test_weights.bin")
    model2 = load_model("test_weights.bin")
    eval_score2 = model2.evaluate_position(features)
    print(f"Loaded model evaluation: {eval_score2:.2f} cp")
    
    # Cleanup
    Path("test_weights.bin").unlink()
