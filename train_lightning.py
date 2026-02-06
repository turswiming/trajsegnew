"""
Training script for trajectories_for_seg using PyTorch Lightning.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import gc
import logging
from datetime import datetime, timezone, timedelta

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from src.losses import FlowSmoothLoss, PointSmoothLoss, TrajectoryLoss_3d
from src.datasets import SegTrainDataset, SegValDataset, DebugTrainDataset, DebugValDataset
from src.utils import (
    setup_logger,
    collate_fn,
    prepare_point_dict,
    calculate_miou,
    save_segmentation_results
)


class SegmentationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for segmentation training.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Set learning rate as hyperparameter for LR finder compatibility
        # This must be set before save_hyperparameters() and must be accessible
        self.learning_rate = config.training.learning_rate
        
        # Save hyperparameters - learning_rate will be included automatically
        self.save_hyperparameters(ignore=['config'])
        
        # Create model
        model_type = getattr(config.model, 'type', 'deltaflow_seg')
        
        if model_type == 'ogc_pointnet':
            from src.models import MaskFormer3D
            n_transformer_layer = getattr(config.model, 'n_transformer_layer', 2)
            transformer_embed_dim = getattr(config.model, 'transformer_embed_dim', 256)
            transformer_input_pos_enc = getattr(config.model, 'transformer_input_pos_enc', False)
            n_point = getattr(config.model, 'n_point', 32768)
            
            self.model = MaskFormer3D(
                n_slot=config.data.num_segments,
                n_point=n_point,
                n_transformer_layer=n_transformer_layer,
                transformer_embed_dim=transformer_embed_dim,
                transformer_input_pos_enc=transformer_input_pos_enc
            )
        elif model_type == 'sparse_unet_seg':
            from src.models import SparseUNetSegModel
            self.model = SparseUNetSegModel(
                voxel_size=getattr(config.model, 'voxel_size', 0.05),
                num_queries=config.data.num_segments,
                maskformer_embed_dim=getattr(config.model, 'transformer_embed_dim', 256),
                maskformer_n_layers=getattr(config.model, 'n_transformer_layer', 2),
            )
        elif model_type == 'deltaflow_seg':
            from src.models import DeltaFlowSeg
            self.model = DeltaFlowSeg(
                voxel_size=getattr(config.model, 'voxel_size', [0.2, 0.2, 0.2]),
                n_slot=config.data.num_segments,
                transformer_embed_dim=getattr(config.model, 'transformer_embed_dim', 256),
                n_transformer_layer=getattr(config.model, 'n_transformer_layer', 2),
            )
        else:
            raise NotImplementedError(f"Model type {model_type} not supported")
        
        # Setup loss functions (will be moved to device in on_train_start)
        self.flow_smooth_loss_fn = None
        self.point_smooth_loss_fn = None
        self.trajectory_loss_fn = None
        self.config = config
        
        # Initialize scheduler attributes
        scheduler_config = getattr(config.training, 'lr_scheduler', {})
        if getattr(scheduler_config, 'type', 'none') == 'cosine_warmup':
            self._warmup_steps = getattr(scheduler_config, 'warmup_steps', 500)
            self._warmup_epochs = getattr(scheduler_config, 'warmup_epochs', 0)
            self._min_lr_ratio = getattr(scheduler_config, 'min_lr_ratio', 0.01)
            self._total_steps = None  # Will be set in on_train_start
        
        # Track best validation loss
        self.best_val_loss = float('inf')
    
    def on_train_start(self):
        """Initialize loss functions on device."""
        # Setup loss functions on device
        self.flow_smooth_loss_fn = FlowSmoothLoss(
            device=self.device,
            **self.config.loss.flow_smooth
        )
        
        self.point_smooth_loss_fn = PointSmoothLoss(
            w_knn=self.config.loss.point_smooth.w_knn,
            w_ball_q=self.config.loss.point_smooth.w_ball_q,
            knn_loss_params=self.config.loss.point_smooth.knn_loss_params,
            ball_q_loss_params=self.config.loss.point_smooth.ball_q_loss_params
        )
        
        # Trajectory loss (optional)
        trajectory_weight = getattr(self.config.loss, 'trajectory_weight', 0.0)
        if trajectory_weight > 0:
            trajectory_config = getattr(self.config.loss, 'trajectory', {})
            self.trajectory_loss_fn = TrajectoryLoss_3d(
                cfg=self.config,
                r=getattr(trajectory_config, 'r', 4),
                criterion=getattr(trajectory_config, 'criterion', 'L2'),
                device=self.device,
                downsample_ratio=self.config.loss.trajectory.downsample_ratio
            )
        else:
            self.trajectory_loss_fn = None
        
        # Set total steps for learning rate scheduler
        if hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
                self._total_steps = self.trainer.estimated_stepping_batches
        
    def forward(self, point_dict):
        return self.model(point_dict)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Initialize loss functions if not done
        if self.flow_smooth_loss_fn is None:
            self.on_train_start()
        
        # Handle batch
        if isinstance(batch['pc'], list):
            pcs_list = [pc.to(self.device) for pc in batch['pc']]
            flows_list = [flow.to(self.device) for flow in batch['flow']]
        else:
            pc = batch['pc'].to(self.device)
            SSL_flow = batch['flow'].to(self.device)
            batch_size = pc.shape[0]
            pcs_list = [pc[i] for i in range(batch_size)]
            flows_list = [SSL_flow[i] for i in range(batch_size)]
        
        batch_size = len(pcs_list)
        
        # Prepare point dictionary
        point_dict = prepare_point_dict(pcs_list, self.device)
        
        # Forward pass
        try:
            outputs = self.model(point_dict)
        except Exception as e:
            logging.warning(f"Error in forward pass: {e}")
            return None
        
        # Get upsampled masks
        student_masks_upsampled = outputs['pred_masks']
        batch_upsampled = outputs['upsampled_batch']
        
        # Split upsampled student masks by batch
        student_masks_list_upsampled = []
        for i in range(batch_size):
            batch_mask = (batch_upsampled == i)
            student_masks_list_upsampled.append(student_masks_upsampled[:, batch_mask])
        
        # Compute losses
        loss = 0.0
        loss_dict = {}
        
        # Flow smooth loss
        flow_smooth_loss = self.flow_smooth_loss_fn(
            pcs_list,
            student_masks_list_upsampled,
            flows_list
        )
        loss += self.config.loss.flow_smooth_weight * flow_smooth_loss
        loss_dict['flow_smooth'] = self.config.loss.flow_smooth_weight * flow_smooth_loss.item()
        
        # Point smooth loss
        if self.config.loss.point_smooth_weight > 0:
            point_smooth_loss = self.point_smooth_loss_fn(
                pcs_list,
                student_masks_list_upsampled
            )
            loss += self.config.loss.point_smooth_weight * point_smooth_loss
            loss_dict['point_smooth'] = self.config.loss.point_smooth_weight * point_smooth_loss.item()
        
        # Trajectory loss
        if self.trajectory_loss_fn is not None and 'trajectories' in batch:
            trajectories_list = batch['trajectories'] if isinstance(batch['trajectories'], list) else [batch['trajectories']]
            
            trajectory_losses = []
            for i in range(batch_size):
                if i >= len(trajectories_list):
                    continue
                
                sample_dict = {
                    'trajectories': trajectories_list[i],
                    'point_cloud': pcs_list[i],
                    'abs_index': 0
                }
                
                mask_i = student_masks_list_upsampled[i]
                if mask_i.shape[1] != pcs_list[i].shape[0]:
                    continue
                
                mask_i_batched = mask_i.unsqueeze(0)
                
                sample_loss = self.trajectory_loss_fn(
                    [sample_dict],
                    [flows_list[i]],
                    mask_i_batched,
                    it=self.global_step,
                    train=True
                )
                trajectory_losses.append(sample_loss)
            
            if len(trajectory_losses) > 0:
                trajectory_loss = sum(trajectory_losses) / len(trajectory_losses)
                trajectory_weight = getattr(self.config.loss, 'trajectory_weight', 0.0)
                if trajectory_weight > 0:
                    loss += trajectory_weight * trajectory_loss
                    loss_dict['trajectory'] = trajectory_loss.item() * trajectory_weight
        
        loss_dict['total'] = loss.item()
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Initialize loss functions if not done
        if self.flow_smooth_loss_fn is None:
            self.on_train_start()
        
        # Handle batch
        if isinstance(batch['pc'], list):
            pcs_list = [pc.to(self.device) for pc in batch['pc']]
            flows_list = [flow.to(self.device) for flow in batch['flow']]
        else:
            pc = batch['pc'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            batch_size = pc.shape[0]
            pcs_list = [pc[i] for i in range(batch_size)]
            flows_list = [gt_flow[i] for i in range(batch_size)]
        
        batch_size = len(pcs_list)
        
        # Prepare point dictionary
        point_dict = prepare_point_dict(pcs_list, self.device)
        outputs = self.model(point_dict)
        
        # Get upsampled student masks
        student_masks_upsampled = outputs['pred_masks']
        upsampled_batch = outputs['upsampled_batch']
        
        # Split upsampled student masks by batch
        student_masks_list_upsampled = []
        for i in range(batch_size):
            batch_mask = (upsampled_batch == i)
            student_masks_list_upsampled.append(student_masks_upsampled[:, batch_mask])
        
        # Compute losses
        if self.config.loss.flow_smooth_weight > 0 and self.flow_smooth_loss_fn is not None:
            flow_smooth_loss = self.flow_smooth_loss_fn(pcs_list, student_masks_list_upsampled, flows_list)
        else:
            flow_smooth_loss = torch.tensor(0.0, device=self.device)
        
        if self.config.loss.point_smooth_weight > 0 and self.point_smooth_loss_fn is not None:
            point_smooth_loss = self.point_smooth_loss_fn(pcs_list, student_masks_list_upsampled)
        else:
            point_smooth_loss = torch.tensor(0.0, device=self.device)
        
        # Calculate mIoU
        miou_list = []
        instance_labels_list = batch['instance_label']
        for i in range(batch_size):
            instance_label = instance_labels_list[i].to(self.device)
            if instance_label.dim() == 1:
                valid_mask = instance_label >= 0
                if valid_mask.sum() == 0:
                    continue
                
                valid_labels = instance_label[valid_mask]
                unique_labels, remapped_labels = torch.unique(valid_labels, return_inverse=True)
                num_gt_instances = len(unique_labels)
                
                if num_gt_instances == 0:
                    continue
                
                full_remapped = torch.full_like(instance_label, -1, dtype=torch.long)
                full_remapped[valid_mask] = remapped_labels.long()
                
                gt_mask_onehot = F.one_hot(
                    torch.clamp(full_remapped, 0, num_gt_instances - 1),
                    num_classes=num_gt_instances
                ).permute(1, 0).float()
                gt_mask_onehot[:, ~valid_mask] = 0.0
            else:
                gt_mask_onehot = instance_label
            
            miou = calculate_miou(student_masks_list_upsampled[i], gt_mask_onehot)
            if miou is not None:
                miou_list.append(miou.item())
        
        # Log validation metrics
        self.log('val/flow_smooth_loss', flow_smooth_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/point_smooth_loss', point_smooth_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size)
        
        if len(miou_list) > 0:
            avg_miou = sum(miou_list) / len(miou_list)
            self.log('val/miou', avg_miou, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return {
            'val_loss': flow_smooth_loss.item(),
            'miou': sum(miou_list) / len(miou_list) if len(miou_list) > 0 else None
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Use self.learning_rate which can be updated by LR finder
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Get scheduler config
        scheduler_config = getattr(self.config.training, 'lr_scheduler', {})
        scheduler_type = getattr(scheduler_config, 'type', 'none')
        
        if scheduler_type == 'cosine_warmup':
            # Create lambda function for learning rate schedule
            def lr_lambda(step):
                # Calculate warmup steps (can be updated in on_train_start)
                warmup_steps = self._warmup_steps
                if self._warmup_epochs > 0 and self._total_steps is not None:
                    steps_per_epoch = self._total_steps / self.trainer.max_epochs
                    warmup_steps = int(self._warmup_epochs * steps_per_epoch)
                
                if step < warmup_steps:
                    # Warmup phase: linear increase from 0 to learning_rate
                    return step / max(warmup_steps, 1)
                else:
                    # Cosine decay phase
                    cosine_step = step - warmup_steps
                    if self._total_steps is not None:
                        cosine_max_steps = self._total_steps - warmup_steps
                        if cosine_max_steps > 0:
                            progress = min(cosine_step / cosine_max_steps, 1.0)
                            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                            return self._min_lr_ratio + (1 - self._min_lr_ratio) * cosine_factor
                    # Fallback: use a simple cosine decay approximation
                    return self._min_lr_ratio + (1 - self._min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * min(cosine_step / 1000, 1)))
            
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                'interval': 'step',  # Update every step
                'frequency': 1,
            }
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        else:
            # No scheduler
            return optimizer
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after training batch ends."""
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save segmentation results every 100 steps
        if self.global_step % 100 == 0 and self.global_step > 0:
            try:
                with torch.no_grad():
                    if isinstance(batch['pc'], list):
                        save_pcs = [pc.to(self.device) for pc in batch['pc']]
                        save_flows = [flow.to(self.device) for flow in batch['flow']]
                    else:
                        save_pcs = [batch['pc'][i].to(self.device) for i in range(batch['pc'].shape[0])]
                        save_flows = [batch['flow'][i].to(self.device) for i in range(batch['flow'].shape[0])]
                    
                    save_point_dict = prepare_point_dict(save_pcs, self.device)
                    save_outputs = self.model(save_point_dict)
                    save_masks = save_outputs['pred_masks']
                    save_batch_tensor = save_outputs['upsampled_batch']
                    
                    save_masks_list = []
                    for i in range(len(save_pcs)):
                        batch_mask = (save_batch_tensor == i)
                        save_masks_list.append(save_masks[:, batch_mask])
                    
                    save_dir = save_segmentation_results(
                        save_pcs, save_masks_list, save_flows,
                        self.config.paths.output_dir, self.global_step
                    )
                    if save_dir:
                        logging.info(f"Saved segmentation results to {save_dir}")
            except Exception as e:
                logging.warning(f"Failed to save segmentation results at step {self.global_step}: {e}")


class SegmentationDataModule(pl.LightningDataModule):
    """Data module for segmentation datasets."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        """Setup datasets."""
        use_debug_datasets = getattr(self.config.data, 'use_debug_datasets', False)
        
        if use_debug_datasets:
            self.train_dataset = DebugTrainDataset(self.config.data.val_dir)
            self.val_dataset = DebugValDataset(self.config.data.val_dir)
        else:
            self.train_dataset = SegTrainDataset(self.config.data.train_dir)
            self.val_dataset = SegValDataset(self.config.data.val_dir)
            
            if len(self.val_dataset) == 0:
                self.val_dataset = SegValDataset(self.config.data.train_dir)
                if len(self.val_dataset) > 0:
                    rng = random.Random(42)
                    indices = list(range(len(self.val_dataset)))
                    rng.shuffle(indices)
                    val_indices = indices[:min(len(indices), 200)]
                    self.val_dataset = Subset(self.val_dataset, val_indices)
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=False,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=False,
            collate_fn=collate_fn
        )


def main(config_path="src/configs/config.yaml"):
    """Main training function."""
    # Load config
    config = OmegaConf.load(config_path)
    
    # Generate timestamp for this run (UTC+8)
    tz_utc8 = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz_utc8).strftime("%Y%m%d_%H%M%S")
    
    # Update paths with timestamp
    config.paths.log_dir = os.path.join(config.paths.log_dir, timestamp)
    config.paths.checkpoint_dir = os.path.join(config.paths.checkpoint_dir, timestamp)
    config.paths.output_dir = os.path.join(config.paths.output_dir, timestamp)
    
    # Create directories
    os.makedirs(config.paths.log_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.output_dir, exist_ok=True)
    
    # Setup logger
    _, logger = setup_logger(config.paths.log_dir)
    logger.info(f"Run timestamp (UTC+8): {timestamp}")
    logger.info("Starting training with PyTorch Lightning...")
    logger.info(f"Log dir: {config.paths.log_dir}")
    logger.info(f"Checkpoint dir: {config.paths.checkpoint_dir}")
    logger.info(f"Output dir: {config.paths.output_dir}")
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Create Lightning module
    model = SegmentationLightningModule(config)
    model = model.to(device)
    
    # Create data module
    data_module = SegmentationDataModule(config)
    data_module.setup()
    
    # Sanity check
    logger.info("Running sanity check on validation set...")
    model.eval()
    sanity_batches = 0
    sanity_max_batches = 2
    sanity_miou_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_module.val_dataloader()):
            if sanity_batches >= sanity_max_batches:
                break
            
            try:
                result = model.validation_step(batch, batch_idx)
                if result and result.get('miou') is not None:
                    sanity_miou_list.append(result['miou'])
                logger.info(f"Sanity check batch {sanity_batches + 1}/{sanity_max_batches}: OK")
                sanity_batches += 1
            except Exception as e:
                logger.error(f"Sanity check failed at batch {sanity_batches + 1}: {e}")
                raise
    
    logger.info(f"Sanity check passed! Processed {sanity_batches} batches successfully.")
    
    if len(sanity_miou_list) == 0:
        raise ValueError("No valid mIoU values calculated during sanity check.")
    
    avg_sanity_miou = sum(sanity_miou_list) / len(sanity_miou_list)
    logger.info(f"Sanity check mIoU: {avg_sanity_miou:.4f}")
    
    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.paths.log_dir,
        name="lightning_logs"
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoint_dir,
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val/flow_smooth_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        accumulate_grad_batches=getattr(config.training, 'gradient_accumulation_steps', 1),
        val_check_interval=getattr(config.training, 'val_check_interval', 0.1),  # Validate frequency (float: epoch fraction, int: batches)
        limit_val_batches=getattr(config.training, 'val_max_batches', 100),
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, data_module)
    
    logger.info("Training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
