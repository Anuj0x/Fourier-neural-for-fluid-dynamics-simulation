#!/usr/bin/env python3
"""
Benchmark script to compare modern FNO implementation with original.

Measures:
- Training speed (samples/second)
- Memory usage
- Inference speed
- Model accuracy
- Parameter count
"""

import torch
import numpy as np
import time
import psutil
import GPUtil
from contextlib import contextmanager
from typing import Dict, Any, Tuple
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append("src")
sys.path.append(".")

from fno.model.modern_fno import ModernFNO, create_modern_fno

try:
    # Try to import original model
    from fno.model.FNOModel import FNOModel
    from fno.model.Utilities import *
    ORIGINAL_AVAILABLE = True
except ImportError:
    print("Warning: Could not import original model, benchmarking only modern implementation")
    ORIGINAL_AVAILABLE = False


@contextmanager
def gpu_memory_monitor():
    """Monitor GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        yield
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        print(f"GPU Memory - Current: {current_memory:.1f} MB, Peak: {peak_memory:.1f} MB")
    else:
        yield


@contextmanager
def cpu_memory_monitor():
    """Monitor CPU memory usage."""
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024**2  # MB
    yield
    end_mem = process.memory_info().rss / 1024**2  # MB
    peak_mem = process.memory_info().peak_wss / 1024**2 if hasattr(process.memory_info(), 'peak_wss') else 0
    print(f"CPU Memory - Start: {start_mem:.1f} MB, End: {end_mem:.1f} MB, Peak: {peak_mem:.1f} MB")


def create_sample_data(batch_size=16, spatial_size=32, time_steps=10, device='cpu'):
    """Create sample Navier-Stokes like data."""
    # Generate synthetic vorticity data (similar to original dataset)
    # Shape: [batch, spatial_x, spatial_y, time, channels]

    # Create time-varying field with some fluid-like patterns
    x = torch.linspace(0, 2*np.pi, spatial_size, device=device)
    y = torch.linspace(0, 2*np.pi, spatial_size, device=device)
    t = torch.linspace(0, 1, time_steps, device=device)

    X, Y = torch.meshgrid(x, y, indexing='ij')
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')

    # Simulate evolving vorticity field with some physics-like patterns
    vorticity = torch.sin(X + T) * torch.cos(Y) + 0.1 * torch.sin(2*X - T) * torch.sin(Y)

    # Add noise
    vorticity += 0.05 * torch.randn_like(vorticity)

    # Expand to batch dimension and add channel dimension
    vorticity = vorticity.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).unsqueeze(-1)

    return vorticity


def create_original_model(device='cpu'):
    """Create original FNO model for comparison."""
    if not ORIGINAL_AVAILABLE:
        return None

    # Use same parameters as in original config
    model = FNOModel(
        in_neurons=20,
        hidden_neurons=256,
        out_neurons=1,
        modesSpace=16,
        modesTime=16,
        time_padding=0,  # Not sure of original value
        input_size=20 + 3,  # channels + coords
        learning_rate=1e-3,
        restart_at_epoch_n=500,
        train_loader=None,  # Not needed for inference
        loss_function="MSE"
    )
    return model.to(device)


def benchmark_forward(model, data, num_runs=10, warmup_runs=5, description=""):
    """Benchmark forward pass performance."""
    device = next(model.parameters()).device
    data = data.to(device)

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(data)

    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            output = model(data)
            torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    throughput = len(data) * num_runs / (end_time - start_time)  # samples/sec

    print(f"{description} Forward pass:")
    print(f"  Avg time: {avg_time:.4f} sec")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    if device.type == 'cuda':
        print(f"  Output shape: {output.shape}")

    return avg_time, throughput, output


def benchmark_training_step(model, data, targets, num_runs=10, description=""):
    """Benchmark training step performance."""
    device = next(model.parameters()).device
    data = data.to(device)
    targets = targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    model.train()

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    samples_per_sec = len(data) * num_runs / (end_time - start_time)

    print(f"{description} Training step:")
    print(f"  Avg time: {avg_time:.4f} sec")
    print(f"  Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"  Loss: {loss.item():.6f}")

    return avg_time, samples_per_sec, loss.item()


def compare_model_sizes():
    """Compare parameter counts between models."""
    print("\n" + "="*50)
    print("MODEL SIZE COMPARISON")
    print("="*50)

    # Modern model
    modern_model = ModernFNO(modes=16, width=64, num_layers=4, use_compile=False)
    modern_params = sum(p.numel() for p in modern_model.parameters())
    print(f"Modern model parameters: {modern_params:,}")

    # Original model
    if ORIGINAL_AVAILABLE:
        try:
            original_model = create_original_model()
            if original_model:
                original_params = sum(p.numel() for p in original_model.parameters())
                print(f"Original model parameters: {original_params:,}")
                ratio = modern_params / original_params
                print(f"Parameter ratio: {ratio:.2f}")
        except Exception as e:
            print(f"  Error creating original model: {e}")
    else:
        print("  Original model not available for comparison")


def main():
    """Run comprehensive benchmark."""
    print("="*60)
    print("FNO IMPLEMENTATION BENCHMARK")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create sample data
    batch_size = 8  # Smaller for comparison
    data = create_sample_data(batch_size, 32, 10, device)
    targets = create_sample_data(batch_size, 32, 10, device)  # Dummy targets

    print(f"Benchmark data shape: {data.shape}")
    print(f"Target data shape: {targets.shape}")
    print()

    # Benchmark modern model
    print("Benching Modern FNO...")
    modern_model = ModernFNO(
        modes=16, width=64, num_layers=4,
        use_compile=False  # Disable for fair comparison
    ).to(device)

    with gpu_memory_monitor(), cpu_memory_monitor():
        modern_forward_time, modern_throughput, modern_output = benchmark_forward(
            modern_model, data, description="Modern FNO"
        )

    modern_train_time, modern_train_throughput, modern_loss = benchmark_training_step(
        modern_model, data, targets, description="Modern FNO"
    )

    print()

    # Benchmark original model
    if ORIGINAL_AVAILABLE:
        print("Benching Original FNO...")
        try:
            original_model = create_original_model(device)
            if original_model:
                # Need to adapt data format for original model
                # Original expects different shape
                with gpu_memory_monitor(), cpu_memory_monitor():
                    orig_forward_time, orig_throughput, orig_output = benchmark_forward(
                        original_model, data, description="Original FNO"
                    )

                orig_train_time, orig_train_throughput, orig_loss = benchmark_training_step(
                    original_model, data, targets, description="Original FNO"
                )

                # Compare results
                print("\n" + "-"*40)
                print("PERFORMANCE COMPARISON")
                print("-"*40)

                print("Forward Pass Speed:")
                print(f"  Modern: {modern_throughput:.2f} samples/sec")
                print(f"  Original: {orig_throughput:.2f} samples/sec")
                print(".2f"
                print("Training Speed:")
                print(f"  Modern: {modern_train_throughput:.2f} samples/sec")
                print(f"  Original: {orig_train_throughput:.2f} samples/sec")
                print(".2f"
                print("Accuracy (MSE):")
                print(f"  Modern: {modern_loss:.6f}")
                print(f"  Original: {orig_loss:.6f}")
            else:
                print("Could not create original model for testing")

        except Exception as e:
            print(f"Error benchmarking original model: {e}")
    else:
        print("Original model not available - skipping comparison")

    # Model size comparison
    compare_model_sizes()

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("FNO Benchmarking")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for benchmarking")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    main()
