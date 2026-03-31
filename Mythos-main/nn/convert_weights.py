#!/usr/bin/env python3
"""
Convert PyTorch model checkpoint to binary weights format for C++ engine.

Usage:
    python convert_weights.py checkpoint.pt weights.bin
    python convert_weights.py --create random_weights.bin
    python convert_weights.py --verify weights.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Import relative to this file's location
sys.path.insert(0, str(Path(__file__).parent))
from nnue_model import NNUEModel


def convert_checkpoint(checkpoint_path: str, output_path: str) -> None:
    """
    Convert PyTorch checkpoint to binary format.
    
    Args:
        checkpoint_path: Path to .pt/.pth checkpoint file
        output_path: Path to output .bin file
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Create model
    model = NNUEModel()
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict)
    
    # Save to binary
    model.save_weights(str(output_path))
    
    print(f"Converted to: {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")


def create_random_weights(output_path: str) -> None:
    """
    Create a random initialized model and save weights.
    
    Args:
        output_path: Path to output .bin file
    """
    print("Creating randomly initialized model...")
    
    model = NNUEModel()
    model.save_weights(output_path)
    
    print(f"Created: {output_path}")


def verify_weights(weights_path: str) -> bool:
    """
    Verify that a weights file is valid and loadable.
    
    Args:
        weights_path: Path to .bin weights file
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(weights_path)
    
    if not path.exists():
        print(f"Error: File not found: {path}")
        return False
    
    print(f"Verifying: {path}")
    print(f"File size: {path.stat().st_size:,} bytes")
    
    try:
        model = NNUEModel()
        model.load_weights(str(path))
        print("✓ Successfully loaded weights")
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        return False
    
    # Test forward pass
    try:
        test_input = torch.zeros(1, 768)
        test_input[0, 0] = 1  # White pawn on A1
        test_input[0, 64 * 5 + 4] = 1  # White king on E1
        test_input[0, 64 * 11 + 60] = 1  # Black king on E8
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Test output: {output.item():.2f} cp")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True


def print_binary_info(weights_path: str) -> None:
    """Print information about a binary weights file."""
    path = Path(weights_path)
    
    if not path.exists():
        print(f"Error: File not found: {path}")
        return
    
    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        print(f"Magic: {magic}")
        
        if magic != b'NNUE':
            print("Warning: Invalid magic number")
            return
        
        version = struct.unpack('<I', f.read(4))[0]
        print(f"Version: {version}")
        
        layer_count = struct.unpack('<I', f.read(4))[0]
        print(f"Layers: {layer_count}")
        
        total_params = 0
        
        for i in range(layer_count):
            in_size = struct.unpack('<I', f.read(4))[0]
            out_size = struct.unpack('<I', f.read(4))[0]
            
            weight_bytes = in_size * out_size * 4
            bias_bytes = out_size * 4
            
            # Read weights for stats
            weight_data = np.frombuffer(f.read(weight_bytes), dtype=np.float32)
            bias_data = np.frombuffer(f.read(bias_bytes), dtype=np.float32)
            
            params = in_size * out_size + out_size
            total_params += params
            
            print(f"\nLayer {i + 1}: {in_size} → {out_size}")
            print(f"  Weights: {weight_data.shape} | mean={weight_data.mean():.6f}, std={weight_data.std():.6f}")
            print(f"  Biases: {bias_data.shape} | mean={bias_data.mean():.6f}, std={bias_data.std():.6f}")
        
        print(f"\nTotal parameters: {total_params:,}")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to binary weights')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert checkpoint to binary')
    convert_parser.add_argument('checkpoint', type=str, help='Input checkpoint path')
    convert_parser.add_argument('output', type=str, help='Output binary path')
    
    # Create command  
    create_parser = subparsers.add_parser('create', help='Create random weights')
    create_parser.add_argument('output', type=str, help='Output binary path')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify weights file')
    verify_parser.add_argument('weights', type=str, help='Weights file to verify')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Print weights file info')
    info_parser.add_argument('weights', type=str, help='Weights file')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        convert_checkpoint(args.checkpoint, args.output)
    elif args.command == 'create':
        create_random_weights(args.output)
    elif args.command == 'verify':
        success = verify_weights(args.weights)
        sys.exit(0 if success else 1)
    elif args.command == 'info':
        print_binary_info(args.weights)
    else:
        # Default behavior for backwards compatibility
        if len(sys.argv) >= 3:
            convert_checkpoint(sys.argv[1], sys.argv[2])
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()
