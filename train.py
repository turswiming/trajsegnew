"""
Training script for trajectories_for_seg.
"""

import os
# Set CUDA memory allocator config before importing torch
# This helps reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os
import numpy as np
from tqdm import tqdm

from src.models import OGCPointNetModel
from src.losses import FlowSmoothLoss, PointSmoothLoss, TrajectoryLoss_3d
from src.datasets import SegTrainDataset, SegValDataset, DebugTrainDataset, DebugValDataset
from src.utils import setup_logger
import torch.nn.functional as F
import gc


def save_segmentation_results(pcs_list, masks_list, flows_list, output_dir, step, batch_idx=0):
    """
    Save point cloud and segmentation results for visualization.
    
    Args:
        pcs_list: List of point clouds [N, 3]
        masks_list: List of mask logits [K, N]
        flows_list: List of flows [N, 3]
        output_dir: Directory to save results
        step: Current training step
        batch_idx: Which sample in batch to save (default: 0)
    """
    save_dir = os.path.join(output_dir, f"step_{step:06d}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save first sample in batch
    if batch_idx < len(pcs_list):
        pc = pcs_list[batch_idx].detach().cpu().numpy()
        mask = masks_list[batch_idx].detach().cpu().numpy()
        flow = flows_list[batch_idx].detach().cpu().numpy()
        
        # Get segmentation labels from mask logits
        seg_labels = np.argmax(mask, axis=0)  # [N]
        
        # Save as npz
        np.savez(
            os.path.join(save_dir, "sample.npz"),
            points=pc,           # [N, 3]
            mask_logits=mask,    # [K, N]
            seg_labels=seg_labels,  # [N]
            flow=flow            # [N, 3]
        )
        
        return save_dir
    return None


def collate_fn(batch):
    """
    Custom collate function for variable-length point clouds.
    Returns lists instead of stacking tensors.
    """
    result = {
        'pc': [item['pc'] for item in batch],
        'flow': [item['flow'] for item in batch]
    }
    
    # Include instance_label if present in any sample
    if 'instance_label' in batch[0]:
        result['instance_label'] = [item['instance_label'] for item in batch]
    
    # Include trajectories if present in any sample
    if 'trajectories' in batch[0]:
        result['trajectories'] = [item['trajectories'] for item in batch]
    
    return result


def prepare_point_dict(pcs_list, device):
    """
    Prepare point dictionary for Sonata model.
    
    Args:
        pcs_list: List of point clouds, each [N_i, 3], or a single tensor [N, 3]
        device: Device to move data to
    
    Returns:
        Dictionary with point data, with correct batch indices for each point cloud
    """
    # Handle both list and single tensor input
    if isinstance(pcs_list, list):
        # Concatenate all point clouds
        all_pcs = torch.cat(pcs_list, dim=0).to(device)
        
        # Create batch indices: each point cloud gets a different batch index
        batch_indices = []
        for batch_idx, pc in enumerate(pcs_list):
            n_points = pc.shape[0] if isinstance(pc, torch.Tensor) else len(pc)
            batch_indices.append(torch.full((n_points,), batch_idx, dtype=torch.long, device=device))
        batch = torch.cat(batch_indices, dim=0)
    else:
        # Single point cloud
        all_pcs = pcs_list.to(device)
        batch = torch.zeros(all_pcs.shape[0], dtype=torch.long, device=device)
    
    coords = all_pcs
    feats = all_pcs  # Use coordinates as features
    
    return {
        'coord': coords,
        'feat': feats,
        'batch': batch
    }


def calculate_miou(pred_mask, gt_mask, min_points=100):
    """
    Calculate Mean Intersection over Union (mIoU) between predicted and ground truth instance masks.
    
    Args:
        pred_mask: Predicted instance masks [K_pred, N] (logits or probabilities)
        gt_mask: Ground truth instance masks [K_gt, N] (one-hot format)
        min_points: Minimum points to consider an instance valid
    
    Returns:
        Mean IoU value, or None if no valid instances found
    """
    # Ensure both masks have the same number of points
    if pred_mask.shape[1] != gt_mask.shape[1]:
        # Truncate or pad to match
        min_n = min(pred_mask.shape[1], gt_mask.shape[1])
        pred_mask = pred_mask[:, :min_n]
        gt_mask = gt_mask[:, :min_n]
    
    # Convert pred_mask to one-hot if it's logits
    pred_mask = pred_mask.contiguous()
    if pred_mask.dim() == 2:
        # Apply softmax if logits
        pred_mask = torch.softmax(pred_mask, dim=0)
        # Convert to one-hot
        pred_mask_argmax = torch.argmax(pred_mask, dim=0)
        # num_classes should match the number of predicted masks
        pred_mask = F.one_hot(pred_mask_argmax, num_classes=pred_mask.shape[0]).permute(1, 0).float()
    
    # Binarize masks
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5
    
    # Calculate sizes
    gt_mask_size = torch.sum(gt_mask, dim=1)
    pred_mask_size = torch.sum(pred_mask, dim=1)
    
    max_iou_list = []
    for j in range(gt_mask.shape[0]):
        if gt_mask_size[j] < min_points:
            continue  # Skip small masks
        
        max_iou = 0.0
        for i in range(pred_mask.shape[0]):
            intersection = torch.sum(pred_mask[i] * gt_mask[j]).float()
            union = pred_mask_size[i] + gt_mask_size[j] - intersection
            iou = intersection / union if union > 0 else 0.0
            
            if not torch.isnan(iou) and iou > max_iou:
                max_iou = iou
        
        max_iou_list.append(max_iou)
    
    if len(max_iou_list) == 0:
        return None
    
    mean_iou = torch.mean(torch.tensor(max_iou_list, device=pred_mask.device, dtype=torch.float32))
    if torch.isnan(mean_iou):
        return None
    return mean_iou


def train_epoch(model, dataloader, optimizer, loss_fns, device, config, logger, tb_writer, epoch, 
                global_step, val_loader=None, val_every_steps=500, val_max_steps=100, best_val_loss=float('inf'),
                gradient_accumulation_steps=1):
    """Train for one epoch with periodic validation.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_fns: Dictionary of loss functions
        device: Device to run on
        config: Config object
        logger: Logger
        tb_writer: TensorBoard writer
        epoch: Current epoch
        global_step: Global training step counter
        val_loader: Validation dataloader (optional)
        val_every_steps: Validate every N steps (default: 500)
        val_max_steps: Maximum validation steps (default: 100)
        best_val_loss: Best validation loss so far
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating (default: 1)
    
    Returns:
        Tuple of (avg_loss, global_step, best_val_loss)
    """
    model.train()
    total_loss = 0.0
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    flow_smooth_loss_fn = loss_fns['flow_smooth']
    point_smooth_loss_fn = loss_fns['point_smooth']
    trajectory_loss_fn = loss_fns.get('trajectory', None)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    accumulation_step = 0
    
    for batch_idx, batch in enumerate(pbar):
        #clear cache
        torch.cuda.empty_cache()
        gc.collect()
        # Handle batch: DataLoader returns list for variable-length tensors
        if isinstance(batch['pc'], list):
            pcs_list = [pc.to(device) for pc in batch['pc']]
            flows_list = [flow.to(device) for flow in batch['flow']]
        else:
            # If batched as tensor, split by batch dimension
            pc = batch['pc'].to(device)
            SSL_flow = batch['flow'].to(device)
            batch_size = pc.shape[0]
            pcs_list = [pc[i] for i in range(batch_size)]
            flows_list = [SSL_flow[i] for i in range(batch_size)]
        
        batch_size = len(pcs_list)
        
        # Prepare point dictionary with correct batch indices
        point_dict = prepare_point_dict(pcs_list, device)
        # Forward pass
        outputs = model(point_dict)
        
        # Get upsampled masks [K, N_total] for flow smooth loss
        student_masks_upsampled = outputs['upsampled_student_masks']  # [K, N_total]
        batch_upsampled = outputs['upsampled_batch']
        # Split upsampled student masks by batch (for flow smooth loss)
        student_masks_list_upsampled = []
        for i in range(batch_size):
            batch_mask = (batch_upsampled == i)
            student_masks_list_upsampled.append(student_masks_upsampled[:, batch_mask])  # [K, N_i]
        
        # Compute losses
        loss = 0.0
        loss_dict = {}
        
        # Flow smooth loss: uses sceneflow as supervision signal
        # Use original point clouds and flows with upsampled student masks
        flow_smooth_loss = flow_smooth_loss_fn(
            pcs_list,  # Original point clouds
            student_masks_list_upsampled,  # Upsampled masks, split by batch
            flows_list  # Original flows
        )
        loss += config.loss.flow_smooth_weight * flow_smooth_loss
        loss_dict['flow_smooth'] = config.loss.flow_smooth_weight * flow_smooth_loss.item()
        
        # Point smooth loss
        if config.loss.point_smooth_weight > 0:
            point_smooth_loss = point_smooth_loss_fn(
                pcs_list,  # Original point clouds
                student_masks_list_upsampled  # Upsampled masks
            )
            loss += config.loss.point_smooth_weight * point_smooth_loss
            loss_dict['point_smooth'] = config.loss.point_smooth_weight * point_smooth_loss.item()
        
        # Trajectory loss (only during training, skip if trajectories not available)
        if trajectory_loss_fn is not None and 'trajectories' in batch:
            trajectories_list = batch['trajectories'] if isinstance(batch['trajectories'], list) else [batch['trajectories']]
            
            # Process each sample separately since point counts may differ
            trajectory_losses = []
            for i in range(batch_size):
                if i >= len(trajectories_list):
                    continue
                
                # Prepare sample dict for this batch item
                sample_dict = {
                    'trajectories': trajectories_list[i],  # (frame_length, N_i, 3)
                    'point_cloud': pcs_list[i],  # (N_i, 3)
                    'abs_index': 0  # Use first frame as reference
                }
                
                # Get mask for this sample: (K, N_i)
                mask_i = student_masks_list_upsampled[i]  # (K, N_i)
                
                # Ensure mask and point cloud have same number of points
                if mask_i.shape[1] != pcs_list[i].shape[0]:
                    continue
                
                # Convert mask from (K, N_i) to (1, K, N_i) for batch dimension
                mask_i_batched = mask_i.unsqueeze(0)  # (1, K, N_i)
                
                # Compute trajectory loss for this sample
                sample_loss = trajectory_loss_fn(
                    [sample_dict],  # List with single sample
                    [flows_list[i]],  # flow parameter (not used but required by interface)
                    mask_i_batched,  # (1, K, N_i)
                    it=global_step,
                    train=True
                )
                trajectory_losses.append(sample_loss)
            
            if len(trajectory_losses) > 0:
                # Average trajectory loss across batch
                trajectory_loss = sum(trajectory_losses) / len(trajectory_losses)
                
                trajectory_weight = getattr(config.loss, 'trajectory_weight', 0.0)
                if trajectory_weight > 0:
                    loss += trajectory_weight * trajectory_loss
                    loss_dict['trajectory'] = trajectory_loss.item() *trajectory_weight
        
        loss_dict['total'] = loss.item()
        total_loss += loss.item()
        
        # Save batch data for saving results before deleting (only if next step will be at 100-step boundary)
        save_batch_data = None
        # Check if this accumulation will complete and result in a step that's a multiple of 100
        will_complete_accumulation = (accumulation_step + 1 >= gradient_accumulation_steps)
        if will_complete_accumulation and (global_step + 1) % 100 == 0:
            # Save batch data for potential saving
            if isinstance(batch['pc'], list):
                save_batch_data = {
                    'pc': [pc.cpu() for pc in batch['pc']],
                    'flow': [flow.cpu() for flow in batch['flow']]
                }
            else:
                save_batch_data = {
                    'pc': batch['pc'].cpu(),
                    'flow': batch['flow'].cpu()
                }
        
        del batch
        del pcs_list
        del flows_list
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        should_validate = False
        try:
            if accumulation_step == 0:
                optimizer.zero_grad()
            loss.backward()
            accumulation_step += 1
            global_step += 1
            # Update weights only after accumulating gradients for specified steps
            if accumulation_step >= gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_step = 0
                should_validate = True  # Mark that we should check for validation
        except Exception as e:
            print(f"Error in backward pass: {e}, not always successful, just let it go")
            accumulation_step = 0
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            should_validate = True
            del student_masks_list_upsampled
            del batch_upsampled
            del student_masks_upsampled
            del point_dict
            del outputs
            continue
        
        # Update progress bar
        pbar.set_postfix(loss=loss.item() * gradient_accumulation_steps, step=global_step)
        
        # Log to tensorboard (log every time global_step is updated, but limit frequency)
        # Log every N steps to avoid too frequent logging
        if global_step % 10 == 0:
            for key, value in loss_dict.items():
                tb_writer.add_scalar(f'train/{key}', value, global_step)
        
        # Save segmentation results every 100 steps (only when step is updated)
        if should_validate and global_step % 100 == 0 and save_batch_data is not None:
            try:
                # Re-run forward pass to get masks for saving
                with torch.no_grad():
                    if isinstance(save_batch_data['pc'], list):
                        save_pcs = [pc.to(device) for pc in save_batch_data['pc']]
                        save_flows = [flow.to(device) for flow in save_batch_data['flow']]
                    else:
                        save_pcs = [save_batch_data['pc'][i].to(device) for i in range(save_batch_data['pc'].shape[0])]
                        save_flows = [save_batch_data['flow'][i].to(device) for i in range(save_batch_data['flow'].shape[0])]
                    save_point_dict = prepare_point_dict(save_pcs, device)
                    save_outputs = model(save_point_dict)
                    save_masks = save_outputs['upsampled_student_masks']
                    save_batch_tensor = save_outputs['upsampled_batch']
                    
                    # Split masks by batch
                    save_masks_list = []
                    for i in range(len(save_pcs)):
                        batch_mask = (save_batch_tensor == i)
                        save_masks_list.append(save_masks[:, batch_mask])
                    
                    save_dir = save_segmentation_results(
                        save_pcs, save_masks_list, save_flows,
                        config.paths.output_dir, global_step
                    )
                    if save_dir:
                        logger.info(f"Saved segmentation results to {save_dir}")
            except Exception as e:
                logger.warning(f"Failed to save segmentation results at step {global_step}: {e}")
        
        # Periodic validation every val_every_steps (only when step is updated and condition is met)
        if should_validate and val_loader is not None and global_step % val_every_steps == 0:
            logger.info(f"\n--- Validation at step {global_step} ---")
            val_loss = validate(model, val_loader, device, config, logger, tb_writer, 
                              global_step, max_steps=val_max_steps)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    config.paths.checkpoint_dir,
                    f"best_step_{global_step}.pth"
                )
                torch.save({
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best checkpoint: {checkpoint_path}")
            
            model.train()  # Ensure model is back in training mode
    
    # Handle remaining accumulated gradients at the end of epoch
    if accumulation_step > 0:
        try:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        except Exception as e:
            print(f"Error in final backward pass: {e}")
            optimizer.zero_grad(set_to_none=True)
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    return avg_loss, global_step, best_val_loss


def validate_batch(model, batch, device, config, loss_fns, calculate_miou_flag=False, logger=None):
    """
    Validate a single batch. Used by both validate() and sanity check.
    
    Args:
        model: Model to validate
        batch: Batch data dictionary
        device: Device to run on
        config: Config object
        loss_fns: Dictionary of loss functions
        calculate_miou_flag: Whether to calculate mIoU
        logger: Logger (optional, for mIoU logging)
    
    Returns:
        Dictionary with losses and optionally mIoU
    """
    # Handle batch: DataLoader returns list for variable-length tensors
    if isinstance(batch['pc'], list):
        pcs_list = [pc.to(device) for pc in batch['pc']]
        flows_list = [flow.to(device) for flow in batch['flow']]
    else:
        # If batched as tensor, split by batch dimension
        pc = batch['pc'].to(device)
        gt_flow = batch['flow'].to(device)
        batch_size = pc.shape[0]
        pcs_list = [pc[i] for i in range(batch_size)]
        flows_list = [gt_flow[i] for i in range(batch_size)]
    
    batch_size = len(pcs_list)
    
    # Prepare point dictionary with correct batch indices
    point_dict = prepare_point_dict(pcs_list, device)
    outputs = model(point_dict)
    
    # Get upsampled student masks and batch tensor (for flow smooth loss)
    student_masks_upsampled = outputs['upsampled_student_masks']  # [K, N_total]
    upsampled_batch = outputs['upsampled_batch']  # [N_total] batch indices for upsampled points
    # Split upsampled student masks by batch using the returned batch tensor
    student_masks_list_upsampled = []
    for i in range(batch_size):
        # Find points belonging to batch i
        batch_mask = (upsampled_batch == i)
        student_masks_list_upsampled.append(student_masks_upsampled[:, batch_mask])  # [K, N_i]
    # Compute losses
    flow_smooth_loss_fn = loss_fns['flow_smooth']
    point_smooth_loss_fn = loss_fns['point_smooth']
    
    # Flow smooth loss
    if config.loss.flow_smooth_weight > 0:
        flow_smooth_loss = flow_smooth_loss_fn(pcs_list, student_masks_list_upsampled, flows_list)
    else:
        flow_smooth_loss = torch.tensor(0.0, device=device)
    
    if config.loss.point_smooth_weight > 0:
        point_smooth_loss = point_smooth_loss_fn(pcs_list, student_masks_list_upsampled)
    else:
        point_smooth_loss = torch.tensor(0.0, device=device)
    
    result = {
        'flow_smooth_loss': flow_smooth_loss.item(),
        'point_smooth_loss': point_smooth_loss.item()
    }
    
    instance_labels_list = batch['instance_label']
    miou_list = []
    for i in range(batch_size):
        instance_label = instance_labels_list[i].to(device)
        if instance_label.dim() == 1:
            # Get actual number of instances in ground truth
            valid_mask = instance_label >= 0
            if valid_mask.sum() == 0:
                if logger:
                    logger.warning(f"  Sample {i}: No valid instance labels (all < 0), skipping mIoU calculation")
                continue
            
            # Remap valid labels to consecutive indices
            valid_labels = instance_label[valid_mask]
            unique_labels, remapped_labels = torch.unique(valid_labels, return_inverse=True)
            num_gt_instances = len(unique_labels)
            
            if num_gt_instances == 0:
                if logger:
                    logger.warning(f"  Sample {i}: No unique instances found, skipping mIoU calculation")
                continue
            
            # Create full one-hot mask
            full_remapped = torch.full_like(instance_label, -1, dtype=torch.long)
            full_remapped[valid_mask] = remapped_labels.long()
            
            # Convert to one-hot
            gt_mask_onehot = F.one_hot(
                torch.clamp(full_remapped, 0, num_gt_instances - 1), 
                num_classes=num_gt_instances
            ).permute(1, 0).float()
            gt_mask_onehot[:, ~valid_mask] = 0.0
        else:
            gt_mask_onehot = instance_label
        
        # Calculate mIoU
        miou = calculate_miou(student_masks_list_upsampled[i], gt_mask_onehot)
        if miou is not None:
            miou_list.append(miou.item())
            if logger:
                logger.info(f"  Sample {i} mIoU: {miou.item():.4f}")
    
    if len(miou_list) > 0:
        result['miou'] = sum(miou_list) / len(miou_list)
    else:
        result['miou'] = None
    
    return result


def validate(model, dataloader, device, config, logger, tb_writer, global_step, max_steps=100):
    """Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device to run on
        config: Config object
        logger: Logger
        tb_writer: TensorBoard writer
        global_step: Global training step for logging
        max_steps: Maximum number of validation steps (default: 100)
    """
    model.eval()
    
    # Setup loss functions
    loss_fns = {
        'flow_smooth': FlowSmoothLoss(
            device=device,
            **config.loss.flow_smooth
        ),
        'point_smooth': PointSmoothLoss(
            w_knn=config.loss.point_smooth.w_knn,
            w_ball_q=config.loss.point_smooth.w_ball_q,
            knn_loss_params=config.loss.point_smooth.knn_loss_params,
            ball_q_loss_params=config.loss.point_smooth.ball_q_loss_params,
        )
    }
    
    total_flow_smooth_loss = 0.0
    total_point_smooth_loss = 0.0
    miou_list = []
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validating (max {max_steps} steps)")):
            if batch_idx >= max_steps:
                break
                
            batch_result = validate_batch(
                model, batch, device, config, loss_fns, 
                calculate_miou_flag=True, logger=None  # Don't log per-sample mIoU
            )
            
            total_flow_smooth_loss += batch_result['flow_smooth_loss']
            total_point_smooth_loss += batch_result['point_smooth_loss']
            num_batches += 1
            
            if 'miou' in batch_result and batch_result['miou'] is not None:
                miou_list.append(batch_result['miou'])
    
    avg_flow_smooth = total_flow_smooth_loss / num_batches
    avg_point_smooth = total_point_smooth_loss / num_batches
    
    logger.info(f"Validation @ step {global_step} ({num_batches} batches) - Flow Smooth: {avg_flow_smooth:.4f}, Point Smooth: {avg_point_smooth:.4f}")
    
    # Log to tensorboard
    tb_writer.add_scalar('val/flow_smooth_loss', avg_flow_smooth, global_step)
    tb_writer.add_scalar('val/point_smooth_loss', avg_point_smooth, global_step)
    
    if len(miou_list) > 0:
        avg_miou = sum(miou_list) / len(miou_list)
        logger.info(f"Validation mIoU: {avg_miou:.4f}")
        tb_writer.add_scalar('val/miou', avg_miou, global_step)
    
    model.train()  # Switch back to training mode
    return avg_flow_smooth


def main(config_path="src/configs/config.yaml"):
    """Main training function."""
    # Load config
    config = OmegaConf.load(config_path)
    
    # Generate timestamp for this run (UTC+8)
    from datetime import datetime, timezone, timedelta
    tz_utc8 = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz_utc8).strftime("%Y%m%d_%H%M%S")
    
    # Update paths with timestamp
    config.paths.log_dir = os.path.join(config.paths.log_dir, timestamp)
    config.paths.checkpoint_dir = os.path.join(config.paths.checkpoint_dir, timestamp)
    config.paths.output_dir = os.path.join(config.paths.output_dir, timestamp)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    os.makedirs(config.paths.log_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.output_dir, exist_ok=True)
    tb_writer, logger = setup_logger(config.paths.log_dir)
    logger.info(f"Run timestamp (UTC+8): {timestamp}")
    logger.info("Starting training...")
    logger.info(f"Log dir: {config.paths.log_dir}")
    logger.info(f"Checkpoint dir: {config.paths.checkpoint_dir}")
    logger.info(f"Output dir: {config.paths.output_dir}")
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")
    
    # Create datasets
    # Check if debug mode is enabled (use same sample from val set repeatedly)
    use_debug_datasets = getattr(config.data, 'use_debug_datasets', False)
    if use_debug_datasets:
        logger.info("Using DEBUG datasets: both train and val will use the same sample from val set")
        debug_train_size = getattr(config.data, 'debug_train_size', 1000)
        debug_val_size = getattr(config.data, 'debug_val_size', 100)
        train_dataset = DebugTrainDataset(config.data.val_dir, dataset_size=debug_train_size)
        val_dataset = DebugValDataset(config.data.val_dir, dataset_size=debug_val_size)
    else:
        train_dataset = SegTrainDataset(config.data.train_dir)
        val_dataset = SegValDataset(config.data.val_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    # Create model - support both TeacherStudentModel and OGCPointNetModel
    model_type = getattr(config.model, 'type', 'teacher_student')  # 'teacher_student' or 'ogc_pointnet'
    
    if model_type == 'ogc_pointnet':
        # OGC PointNet++ model
        logger.info("Using OGC PointNet++ model")
        n_transformer_layer = getattr(config.model, 'n_transformer_layer', 2)
        transformer_embed_dim = getattr(config.model, 'transformer_embed_dim', 256)
        transformer_input_pos_enc = getattr(config.model, 'transformer_input_pos_enc', False)
        n_point = getattr(config.model, 'n_point', 32768)
        model = OGCPointNetModel(
            num_queries=config.data.num_segments,
            n_point=n_point,  # Will be adjusted based on input
            n_transformer_layer=n_transformer_layer,
            transformer_embed_dim=transformer_embed_dim,
            transformer_input_pos_enc=transformer_input_pos_enc
        ).to(device)
    else:
        # Teacher-Student architecture with MaskFormer
        logger.info("Using Teacher-Student model")
        pretrained_path = getattr(config.model, 'pretrained_path', None)
        freeze_encoder = getattr(config.model, 'freeze_encoder', False)
        
        model = TeacherStudentModel(
            use_pretrained=config.model.use_pretrained,
            pretrained_name=config.model.pretrained_name,
            pretrained_path=pretrained_path,
            feature_dim=config.model.feature_dim,
            num_queries=config.data.num_segments,  # Number of mask queries
            ema_decay=config.model.ema_decay,
            freeze_encoder=freeze_encoder
        ).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Setup loss functions
    loss_fns = {
        'flow_smooth': FlowSmoothLoss(
            device=device,
            **config.loss.flow_smooth
        ),
        'point_smooth': PointSmoothLoss(
            w_knn=config.loss.point_smooth.w_knn,
            w_ball_q=config.loss.point_smooth.w_ball_q,
            knn_loss_params=config.loss.point_smooth.knn_loss_params,
            ball_q_loss_params=config.loss.point_smooth.ball_q_loss_params
        )
    }
    
    # Setup trajectory loss if configured
    trajectory_weight = getattr(config.loss, 'trajectory_weight', 0.0)
    if trajectory_weight > 0:
        trajectory_config = getattr(config.loss, 'trajectory', {})
        trajectory_loss_fn = TrajectoryLoss_3d(
            cfg=config,
            r=getattr(trajectory_config, 'r', 4),
            criterion=getattr(trajectory_config, 'criterion', 'L2'),
            device=device,
            downsample_ratio=config.loss.trajectory.downsample_ratio
        )
        loss_fns['trajectory'] = trajectory_loss_fn
        logger.info(f"Trajectory loss enabled with weight {trajectory_weight}")
    else:
        loss_fns['trajectory'] = None
    
    # Sanity check: run a few batches on validation set to verify everything works and calculate mIoU
    logger.info("Running sanity check on validation set...")
    model.eval()
    sanity_batches = 0
    sanity_max_batches = 2
    sanity_miou_list = []
    
    # Setup loss functions for sanity check (same as validation)
    loss_fns_sanity = {
        'flow_smooth': FlowSmoothLoss(
            device=device,
            **config.loss.flow_smooth
        ),
        'point_smooth': PointSmoothLoss(
            w_knn=config.loss.point_smooth.w_knn,
            w_ball_q=config.loss.point_smooth.w_ball_q,
            knn_loss_params=config.loss.point_smooth.knn_loss_params,
            ball_q_loss_params=config.loss.point_smooth.ball_q_loss_params
        )
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if sanity_batches >= sanity_max_batches:
                break
            
            try:
                # Use the same validate_batch function as validation
                batch_result = validate_batch(
                    model, batch, device, config, loss_fns_sanity,
                    calculate_miou_flag=True, logger=logger
                )
                
                logger.info(f"Sanity check batch {sanity_batches + 1}/{sanity_max_batches}: OK")
                logger.info(f"  Flow Smooth Loss: {batch_result['flow_smooth_loss']:.4f}")
                
                if 'miou' in batch_result and batch_result['miou'] is not None:
                    sanity_miou_list.append(batch_result['miou'])
                    logger.info(f"  Batch mIoU: {batch_result['miou']:.4f}")
                
                sanity_batches += 1
            except Exception as e:
                logger.error(f"Sanity check failed at batch {sanity_batches + 1}: {e}")
                raise
            
            # Clear cache
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    logger.info(f"Sanity check passed! Processed {sanity_batches} batches successfully.")
    
    # mIoU is required
    if len(sanity_miou_list) == 0:
        raise ValueError("No valid mIoU values calculated during sanity check. Please check that instance_label is provided and contains valid instances.")
    
    avg_sanity_miou = sum(sanity_miou_list) / len(sanity_miou_list)
    logger.info(f"Sanity check mIoU: {avg_sanity_miou:.4f} (from {len(sanity_miou_list)} samples)")
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    val_every_steps = 100  # Validate every 500 steps
    val_max_steps = 100    # Max 100 validation steps
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    
    logger.info(f"Training with validation every {val_every_steps} steps, max {val_max_steps} validation batches")
    
    # Get gradient accumulation steps from config, default to 1
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    for epoch in range(config.training.num_epochs):
        # Train with periodic validation
        train_loss, global_step, best_val_loss = train_epoch(
            model, train_loader, optimizer, loss_fns,
            device, config, logger, tb_writer, epoch,
            global_step=global_step,
            val_loader=val_loader,
            val_every_steps=val_every_steps,
            val_max_steps=val_max_steps,
            best_val_loss=best_val_loss,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Save periodic checkpoint at end of epoch
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint_path = os.path.join(
                config.paths.checkpoint_dir,
                f"checkpoint_epoch_{epoch}_step_{global_step}.pth"
            )
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved periodic checkpoint: {checkpoint_path}")
    
    logger.info("Training completed!")
    tb_writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/config.yaml")
    args = parser.parse_args()
    main(args.config)

