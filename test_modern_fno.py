#!/usr/bin/env python3
"""
Quick test script for the modernized FNO implementation.
"""

import torch
import sys
import os

# Add src to path
sys.path.append("src")
sys.path.append(".")

from fno.model.modern_fno import ModernFNO

def test_modern_fno():
    """Test the modern FNO implementation."""
    print("Testing Modern FNO Implementation")
    print("="*40)

    # Create model
    model = ModernFNO(
        modes=16,
        width=64,
        num_layers=4,
        use_compile=False  # Disable for testing
    )

    # Create sample data (Navier-Stokes vorticity)
    batch_size, spatial_size, time_steps = 4, 32, 10
    x = torch.randn(batch_size, spatial_size, spatial_size, time_steps, 1)

    print(f"Input shape: {x.shape}")

    # Test forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")

    # Check shapes
    expected_shape = (batch_size, spatial_size, spatial_size, time_steps, 1)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    # Test parameter counts
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    print("\n✓ All tests passed!")

    return model

def test_training_step():
    """Test a training step."""
    print("\nTesting Training Step")
    print("-" * 30)

    model = ModernFNO(modes=8, width=32, num_layers=2, use_compile=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Fake data
    x = torch.randn(2, 32, 32, 10, 1)
    y = torch.randn(2, 32, 32, 10, 1)

    model.train()
    optimizer.zero_grad()

    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    print(f"Training loss: {loss.item():.6f}")
    print("✓ Training step successful!")

if __name__ == "__main__":
    test_modern_fno()
    test_training_step()
