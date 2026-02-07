"""
Learning Rate Finder script using PyTorch Lightning.

This script helps find the optimal learning rate for training.
"""

import os
import torch
import argparse
import random
import numpy as np
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


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_lr_finder(config_path, min_lr, max_lr, num_training, seed, batch_size=None, accumulate_grad_batches=1, return_lr_finder=False):
    """
    Run a single LR finder iteration.
    
    Returns:
        If return_lr_finder=False: suggested_lr (float or None)
        If return_lr_finder=True: (suggested_lr, lr_finder) tuple or (None, None) if failed
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Temporarily disable learning rate scheduler for LR finder
    # LR finder needs to control learning rate itself, so warmup/cosine decay should not interfere
    original_scheduler_type = None
    if hasattr(config.training, 'lr_scheduler'):
        original_scheduler_type = getattr(config.training.lr_scheduler, 'type', None)
        config.training.lr_scheduler.type = 'none'  # Disable scheduler for LR finder
    
    # Override batch size if specified
    if batch_size is not None:
        config.training.batch_size = batch_size
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SegmentationLightningModule(config)
    model = model.to(device)
    
    # Create data module
    data_module = SegmentationDataModule(config)
    data_module.setup()
    
    # Create trainer for LR finder
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,  # Disable logging for individual runs
        enable_progress_bar=False,  # Disable progress bar for cleaner output
        enable_model_summary=False,
        deterministic=True,
        accumulate_grad_batches=accumulate_grad_batches,  # Gradient accumulation to reduce memory usage
    )
    
    # Run LR Finder
    try:
        if Tuner is not None:
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(
                model,
                datamodule=data_module,
                min_lr=min_lr,
                max_lr=max_lr,
                num_training=num_training,
            )
        else:
            if hasattr(trainer, 'tuner') and trainer.tuner is not None:
                lr_finder = trainer.tuner.lr_find(
                    model,
                    datamodule=data_module,
                    min_lr=min_lr,
                    max_lr=max_lr,
                    num_training=num_training,
                )
            else:
                return (None, None) if return_lr_finder else None
        
        suggested_lr = lr_finder.suggestion()
        if return_lr_finder:
            return suggested_lr, lr_finder
        else:
            return suggested_lr
        
    except Exception as e:
        print(f"Warning: LR finder failed with seed {seed}: {e}")
        return (None, None) if return_lr_finder else None


def main(config_path, min_lr=1e-6, max_lr=1e-1, num_training=300, save_plot=False, seed=42, num_runs=1, batch_size=None, accumulate_grad_batches=1):
    """
    Run Learning Rate Finder to find optimal learning rate.
    
    Args:
        config_path: Path to config file
        min_lr: Minimum learning rate to test (default: 1e-6)
        max_lr: Maximum learning rate to test (default: 1e-1)
        num_training: Number of training steps for LR finder (default: 300)
        save_plot: Whether to save the LR finder plot (default: False)
        seed: Random seed for reproducibility (default: 42)
        num_runs: Number of runs to average over (default: 1)
        batch_size: Override batch size for LR finder (default: None, uses config value)
        accumulate_grad_batches: Number of batches to accumulate gradients (default: 1). 
                                 Increase to reduce memory usage, e.g., 2 or 4
    """
    # Load config
    config = OmegaConf.load(config_path)
    
    # Note: Learning rate scheduler (including warmup) will be temporarily disabled
    # during LR finder runs to avoid interference. The scheduler is only disabled
    # during the LR finder execution, not permanently modified.
    
    # Generate timestamp for this run (UTC+8)
    tz_utc8 = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz_utc8).strftime("%Y%m%d_%H%M%S")
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if batch_size is not None:
        print(f"Using batch size: {batch_size} (overriding config value: {config.training.batch_size})")
    else:
        print(f"Using batch size from config: {config.training.batch_size}")
    
    if accumulate_grad_batches > 1:
        effective_batch_size = (batch_size if batch_size else config.training.batch_size) * accumulate_grad_batches
        print(f"Gradient accumulation: {accumulate_grad_batches} batches")
        print(f"Effective batch size: {effective_batch_size} (batch_size × accumulate_grad_batches)")
    
    print("\n" + "="*60)
    print("Running Learning Rate Finder...")
    print(f"Learning rate range: [{min_lr:.2e}, {max_lr:.2e}]")
    print(f"Number of training steps per run: {num_training}")
    print(f"Number of runs: {num_runs}")
    print(f"Random seed(s): {seed}" + (f" to {seed + num_runs - 1}" if num_runs > 1 else ""))
    print("="*60 + "\n")
    
    # Run LR Finder multiple times if requested
    suggested_lrs = []
    lr_finder_obj = None  # Store lr_finder object for plotting
    
    for run_idx in range(num_runs):
        current_seed = seed + run_idx
        print(f"Run {run_idx + 1}/{num_runs} (seed: {current_seed})...")
        
        # For first run and if save_plot is requested, also return lr_finder object
        return_lr_finder = (run_idx == 0 and save_plot)
        
        result = run_single_lr_finder(
            config_path=config_path,
            min_lr=min_lr,
            max_lr=max_lr,
            num_training=num_training,
            seed=current_seed,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            return_lr_finder=return_lr_finder
        )
        
        if return_lr_finder:
            suggested_lr, lr_finder_obj = result
        else:
            suggested_lr = result
        
        if suggested_lr is not None:
            suggested_lrs.append(suggested_lr)
            print(f"  Suggested LR: {suggested_lr:.6e}")
        else:
            print(f"  Run {run_idx + 1} failed, skipping...")
    
    if len(suggested_lrs) == 0:
        print("\n❌ All LR finder runs failed!")
        return None
    
    # Calculate statistics
    suggested_lrs = np.array(suggested_lrs)
    mean_lr = np.mean(suggested_lrs)
    median_lr = np.median(suggested_lrs)
    std_lr = np.std(suggested_lrs)
    
    # Use median as final suggestion (more robust to outliers)
    final_suggested_lr = median_lr
    
    print("\n" + "="*60)
    print("LR Finder Results:")
    print("="*60)
    if len(suggested_lrs) > 1:
        print(f"Number of successful runs: {len(suggested_lrs)}")
        print(f"Suggested learning rates:")
        for i, lr in enumerate(suggested_lrs):
            print(f"  Run {i+1}: {lr:.6e}")
        print(f"\nStatistics:")
        print(f"  Mean:   {mean_lr:.6e}")
        print(f"  Median: {median_lr:.6e} (used as final suggestion)")
        print(f"  Std:    {std_lr:.6e}")
        print(f"  Range:  [{np.min(suggested_lrs):.6e}, {np.max(suggested_lrs):.6e}]")
    print(f"\nFinal suggested learning rate: {final_suggested_lr:.6e}")
    print(f"Final suggested learning rate: {final_suggested_lr:.2e}")
    print("="*60)
    
    # Save plot if requested (use lr_finder object from first run)
    if save_plot and lr_finder_obj is not None:
        print(f"\nSaving LR finder plot...")
        try:
            plot_path = "lr_finder_plot.png"
            fig = lr_finder_obj.plot(suggest=True, show=False)
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to save plot: {e}")
    
    # Print current config learning rate for comparison
    current_lr = config.training.learning_rate
    print(f"\nCurrent config learning rate: {current_lr:.6e}")
    if final_suggested_lr:
        ratio = final_suggested_lr / current_lr
        print(f"Ratio (suggested / current): {ratio:.2f}x")
        if ratio > 2.0:
            print("⚠️  Suggested LR is significantly higher than current LR")
        elif ratio < 0.5:
            print("⚠️  Suggested LR is significantly lower than current LR")
        else:
            print("✓ Suggested LR is close to current LR")
    
    return final_suggested_lr


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
        default=1e-8,
        help="Minimum learning rate to test (default: 1e-6)"
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=1e-2,
        help="Maximum learning rate to test (default: 1e-1)"
    )
    parser.add_argument(
        "--num_training",
        type=int,
        default=100,
        help="Number of training steps for LR finder (default: 300, increased for better stability)"
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Save LR finder plot to file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). If num_runs > 1, seeds will be seed, seed+1, ..., seed+num_runs-1"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs to average over (default: 1). Use 3-5 for more stable results when loss varies greatly between samples"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for LR finder (default: None, uses config value). Increase to reduce impact of individual samples"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=2,
        help="Number of batches to accumulate gradients before updating (default: 1). "
             "Increase to reduce memory usage, e.g., 2 or 4. Effective batch size = batch_size × accumulate_grad_batches"
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_training=args.num_training,
        save_plot=args.save_plot,
        seed=args.seed,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches
    )
