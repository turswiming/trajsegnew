"""
Learning Rate Finder script using PyTorch Lightning.

This script helps find the optimal learning rate for training.
"""

import os
import torch
import argparse
from omegaconf import OmegaConf
from datetime import datetime, timezone, timedelta

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# Try to import Tuner - handle different PyTorch Lightning versions
Tuner = None
try:
    from pytorch_lightning.tuner.tuning import Tuner
except ImportError:
    try:
        from pytorch_lightning.tuner import Tuner
    except ImportError:
        try:
            # Try lightning.pytorch path (for newer versions)
            from lightning.pytorch.tuner.tuning import Tuner
        except ImportError:
            try:
                from lightning.pytorch.tuner import Tuner
            except ImportError:
                # For older versions, Tuner might not exist
                Tuner = None

# Import classes from train_lightning.py
from train_lightning import SegmentationLightningModule, SegmentationDataModule


def main(config_path, min_lr=1e-6, max_lr=1e-1, num_training=100, save_plot=False):
    """
    Run Learning Rate Finder to find optimal learning rate.
    
    Args:
        config_path: Path to config file
        min_lr: Minimum learning rate to test (default: 1e-6)
        max_lr: Maximum learning rate to test (default: 1e-1)
        num_training: Number of training steps for LR finder (default: 100)
        save_plot: Whether to save the LR finder plot (default: False)
    """
    # Load config
    config = OmegaConf.load(config_path)
    
    # Generate timestamp for this run (UTC+8)
    tz_utc8 = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz_utc8).strftime("%Y%m%d_%H%M%S")
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = SegmentationLightningModule(config)
    model = model.to(device)
    
    # Create data module
    print("Setting up data module...")
    data_module = SegmentationDataModule(config)
    data_module.setup()
    
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Val dataset size: {len(data_module.val_dataset)}")
    
    # Setup TensorBoard logger for LR finder with timestamp
    tb_logger = TensorBoardLogger(
        save_dir=config.paths.log_dir,
        name="lr_finder",
        version=timestamp
    )
    print(f"TensorBoard logs will be saved to: {tb_logger.log_dir}")
    
    # Create trainer for LR finder
    # Note: We don't set max_epochs because LR finder will run for a fixed number of steps
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=tb_logger,  # Enable TensorBoard logging
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    print("\n" + "="*60)
    print("Running Learning Rate Finder...")
    print(f"Learning rate range: [{min_lr:.2e}, {max_lr:.2e}]")
    print(f"Number of training steps: {num_training}")
    print("="*60 + "\n")
    
    # Run LR Finder
    try:
        # Handle different PyTorch Lightning versions
        if Tuner is not None:
            # PyTorch Lightning 2.0+: Create Tuner instance explicitly
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(
                model,
                datamodule=data_module,
                min_lr=min_lr,
                max_lr=max_lr,
                num_training=num_training,
            )
        else:
            # Older versions: Use trainer.tuner (if available)
            if hasattr(trainer, 'tuner') and trainer.tuner is not None:
                lr_finder = trainer.tuner.lr_find(
                    model,
                    datamodule=data_module,
                    min_lr=min_lr,
                    max_lr=max_lr,
                    num_training=num_training,
                )
            else:
                raise RuntimeError(
                    "LR Finder not available. Please upgrade to PyTorch Lightning 2.0+ "
                    "or ensure tuner is properly initialized."
                )
        
        # Get suggested learning rate
        suggested_lr = lr_finder.suggestion()
        
        print("\n" + "="*60)
        print("LR Finder Results:")
        print("="*60)
        print(f"Suggested learning rate: {suggested_lr:.6e}")
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        print("="*60)
        
        # Save plot if requested
        if save_plot:
            plot_path = "lr_finder_plot.png"
            print(f"\nSaving LR finder plot to {plot_path}...")
            fig = lr_finder.plot(suggest=True, show=False)
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        # Print current config learning rate for comparison
        current_lr = config.training.learning_rate
        print(f"\nCurrent config learning rate: {current_lr:.6e}")
        if suggested_lr:
            ratio = suggested_lr / current_lr
            print(f"Ratio (suggested / current): {ratio:.2f}x")
            if ratio > 2.0:
                print("⚠️  Suggested LR is significantly higher than current LR")
            elif ratio < 0.5:
                print("⚠️  Suggested LR is significantly lower than current LR")
            else:
                print("✓ Suggested LR is close to current LR")
        
        return suggested_lr
        
    except Exception as e:
        print(f"\n❌ Error during LR Finder: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal learning rate using PyTorch Lightning")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate to test (default: 1e-6)"
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=1e-1,
        help="Maximum learning rate to test (default: 1e-1)"
    )
    parser.add_argument(
        "--num_training",
        type=int,
        default=100,
        help="Number of training steps for LR finder (default: 100)"
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Save LR finder plot to file"
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_training=args.num_training,
        save_plot=args.save_plot
    )
