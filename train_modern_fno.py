#!/usr/bin/env python3
"""
Modern training script for Fourier Neural Operator.

Key improvements over original:
- Production-ready structure (not Jupyter notebook based)
- Modern PyTorch Lightning practices
- Mixed precision training
- Better checkpointing and logging
- Hyperparameter sweeping support
- Distributed training support
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    DeviceStatsMonitor, RichModelSummary
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch
import numpy as np
from pathlib import Path

from src.fno.model.modern_fno import ModernFNO, create_modern_fno


class NavierStokesDataModule(pl.LightningDataModule):
    """Modern data module with optimized loading and preprocessing."""

    def __init__(self, data_path: str, batch_size: int = 16, num_workers: int = 4,
                 train_split: float = 0.8, normalize: bool = True):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.normalize = normalize

    def setup(self, stage: str = None):
        """Load and preprocess data."""
        # Load data (maintain compatibility with original)
        data = np.load(self.data_path)
        if self.normalize:
            data = self._normalize_data(data)

        # Split into train/val
        n_samples = data.shape[0]
        train_size = int(self.train_split * n_samples)

        self.train_data = torch.from_numpy(data[:train_size]).float()
        self.val_data = torch.from_numpy(data[train_size:]).float()

        print(f"Train samples: {len(self.train_data)}, Val samples: {len(self.val_data)}")

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [-1, 1] range."""
        # Simple min-max normalization
        data_min, data_max = data.min(), data.max()
        return 2 * (data - data_min) / (data_max - data_min) - 1

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            persistent_workers=True, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            persistent_workers=True, pin_memory=True
        )


def create_model_checkpoint_callback(save_dir: str, monitor_metric: str = 'val_loss'):
    """Create advanced checkpointing."""
    return ModelCheckpoint(
        dirpath=save_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        monitor=monitor_metric,
        mode='min',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=True
    )


def create_callbacks(save_dir: str, patience: int = 20):
    """Create training callbacks."""
    return [
        create_model_checkpoint_callback(save_dir),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
        DeviceStatsMonitor(cpu_stats=True),
        RichModelSummary(max_depth=2)
    ]


@hydra.main(version_base=None, config_path="configs", config_name="modern_fno_train")
def train_fno(cfg: DictConfig):
    """Main training function with Hydra configuration."""

    # Set random seed
    pl.seed_everything(cfg.training.seed)

    # Setup data module
    data_module = NavierStokesDataModule(
        data_path=cfg.data.data_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        normalize=cfg.data.normalize
    )

    # Setup model
    if cfg.model.use_original_api:
        # Backward compatibility
        model = create_modern_fno(
            in_neurons=cfg.model.in_neurons,
            hidden_neurons=cfg.model.hidden_neurons,
            out_neurons=cfg.model.out_neurons,
            modes_space=cfg.model.modes_space,
            modes_time=cfg.model.modes_time,
            learning_rate=cfg.training.learning_rate,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            use_compile=cfg.model.use_compile
        )
    else:
        model = ModernFNO(
            modes=cfg.model.modes,
            width=cfg.model.width,
            num_layers=cfg.model.num_layers,
            learning_rate=cfg.training.learning_rate,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            use_compile=cfg.model.use_compile
        )

    # Setup logging
    loggers = []
    if cfg.logging.use_wandb:
        loggers.append(WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name,
            save_dir=cfg.logging.save_dir
        ))
    else:
        loggers.append(TensorBoardLogger(
            save_dir=cfg.logging.save_dir,
            name="modern_fno"
        ))

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=create_callbacks(cfg.logging.save_dir, cfg.training.patience),
        logger=loggers,
        enable_progress_bar=cfg.logging.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        num_sanity_val_steps=cfg.training.num_sanity_val_steps,
    )

    # Train model
    trainer.fit(model, datamodule=data_module)

    # Test best model
    if cfg.testing.test_after_training:
        trainer.test(model, datamodule=data_module)

    print(f"Training completed! Best model saved at: {trainer.checkpoint_callback.best_model_path}")

    return trainer.checkpoint_callback.best_model_path


if __name__ == "__main__":
    # Quick test/run
    import argparse

    parser = argparse.ArgumentParser("Train Modern Fourier Neural Operator")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode (don't use wandb, minimal epochs)")

    args = parser.parse_args()

    if args.test:
        # Quick test configuration
        print("Running in test mode...")

        # Create minimal test data
        test_data_path = "src/data/datasets/test_data.npy"
        if not os.path.exists(test_data_path):
            print("Creating test data...")
            # Generate small synthetic dataset
            test_data = np.random.randn(100, 32, 32, 10, 1).astype(np.float32)
            np.save(test_data_path, test_data)

        # Minimal configuration for testing
        test_config = {
            "data": {
                "data_path": test_data_path,
                "normalize": True
            },
            "model": {
                "modes": 4,  # Small for quick testing
                "width": 32,
                "num_layers": 2,
                "learning_rate": 1e-3,
                "activation": "gelu",
                "dropout": 0.0,
                "use_compile": False  # Disable for quick iteration
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 2,
                "accelerator": "auto",
                "devices": "auto",
                "precision": "32",
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 1,
                "patience": 10,
                "val_check_interval": 1.0,
                "num_sanity_val_steps": 0,
                "seed": 42
            },
            "logging": {
                "use_wandb": False,
                "save_dir": "./checkpoints",
                "enable_progress_bar": True,
                "log_every_n_steps": 1
            },
            "testing": {
                "test_after_training": False
            }
        }

        # Convert to config and train
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(test_config)
        best_checkpoint = train_fno(cfg)

    else:
        # Full training with hydra
        train_fno()
