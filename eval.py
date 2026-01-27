"""
Evaluation script for trajectories_for_seg.
"""

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os
import argparse

from src.models import TeacherStudentSegmentationModel
from src.datasets import SegValDataset
from src.utils import setup_logger
from src.losses import FlowSmoothLoss, PointSmoothLoss
from src.utils.upsampling import upsample_masks_from_point
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """Consistency loss between student and teacher segmentation predictions."""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, student_masks, teacher_masks):
        """Compute consistency loss between student and teacher masks."""
        return self.criterion(student_masks, teacher_masks.detach())


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
        pred_mask: Predicted instance masks [K, N] (logits or probabilities)
        gt_mask: Ground truth instance masks [K, N] (one-hot format)
        min_points: Minimum points to consider an instance valid
    
    Returns:
        Mean IoU value, or None if no valid instances found
    """
    # Convert pred_mask to one-hot if it's logits
    if pred_mask.dim() == 2:
        # Apply softmax if logits
        pred_mask = torch.softmax(pred_mask, dim=0)
        # Convert to one-hot
        pred_mask_argmax = torch.argmax(pred_mask, dim=0)
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


def evaluate(model, dataloader, device, config, logger):
    """Evaluate model."""
    model.eval()
    total_flow_smooth_loss = 0.0
    total_point_smooth_loss = 0.0
    total_consistency_loss = 0.0
    miou_list = []
    num_samples = 0
    
    consistency_loss_fn = ConsistencyLoss()
    flow_smooth_loss_fn = FlowSmoothLoss(
        device=device,
        **config.loss.flow_smooth
    )
    point_smooth_loss_fn = PointSmoothLoss(
        w_knn=config.loss.point_smooth.w_knn,
        w_ball_q=config.loss.point_smooth.w_ball_q,
        knn_loss_params=config.loss.point_smooth.knn_loss_params,
        ball_q_loss_params=config.loss.point_smooth.ball_q_loss_params
    )
    
    with torch.no_grad():
        for batch in dataloader:
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
            outputs = model(point_dict, return_teacher=True)
            student_masks_encoded = outputs['student_segmentation_masks']  # [K, N_encoded]
            teacher_masks_encoded = outputs['teacher_segmentation_masks']  # [K, N_encoded]
            
            # Get Point objects for upsampling
            student_features = outputs['student_features']
            total_original_points = sum(pc.shape[0] for pc in pcs_list)
            
            # Upsample masks to original point cloud scale
            student_masks_upsampled = upsample_masks_from_point(
                student_features,
                student_masks_encoded,
                total_original_points
            )
            teacher_masks_upsampled = upsample_masks_from_point(
                student_features,
                teacher_masks_encoded,
                total_original_points
            )
            
            # Split masks by batch
            student_masks_list = []
            teacher_masks_list = []
            start_idx = 0
            for i in range(batch_size):
                n_points = pcs_list[i].shape[0]
                end_idx = start_idx + n_points
                
                student_mask_i = student_masks_upsampled[:, start_idx:end_idx]
                teacher_mask_i = teacher_masks_upsampled[:, start_idx:end_idx]
                student_masks_list.append(student_mask_i)
                teacher_masks_list.append(teacher_mask_i)
                
                start_idx = end_idx
            
            # Compute evaluation losses
            consistency_loss = consistency_loss_fn(student_masks_upsampled, teacher_masks_upsampled)
            flow_smooth_loss = flow_smooth_loss_fn(pcs_list, student_masks_list, flows_list)
            point_smooth_loss = point_smooth_loss_fn(pcs_list, student_masks_list)
            
            total_consistency_loss += consistency_loss.item()
            total_flow_smooth_loss += flow_smooth_loss.item()
            total_point_smooth_loss += point_smooth_loss.item()
            
            # Calculate mIoU - instance_label is required
            if 'instance_label' not in batch:
                raise ValueError("instance_label is required for evaluation but not found in batch. Please ensure the dataset provides ground truth instance labels.")
            
            instance_labels_list = batch['instance_label']
            if not isinstance(instance_labels_list, list):
                raise ValueError(f"instance_label should be a list, but got {type(instance_labels_list)}")
            
            for i in range(batch_size):
                if i >= len(instance_labels_list):
                    raise ValueError(f"instance_label list has {len(instance_labels_list)} elements but batch_size is {batch_size}")
                
                # Convert instance_label to one-hot format if needed
                instance_label = instance_labels_list[i].to(device)
                if instance_label.dim() == 1:
                    # Get actual number of instances in ground truth
                    # instance_label values should be in range [0, max_instance_id]
                    # We need to remap them to consecutive indices [0, 1, 2, ...]
                    unique_labels, remapped_labels = torch.unique(instance_label, return_inverse=True)
                    num_gt_instances = len(unique_labels)
                    
                    # Convert to one-hot: [num_gt_instances, N]
                    gt_mask_onehot = F.one_hot(remapped_labels.long(), num_classes=num_gt_instances).permute(1, 0).float()
                else:
                    gt_mask_onehot = instance_label
                
                # Calculate mIoU for this sample
                miou = calculate_miou(student_masks_list[i], gt_mask_onehot)
                if miou is not None:
                    miou_list.append(miou.item())
            
            num_samples += 1
    
    avg_consistency = total_consistency_loss / num_samples
    avg_flow_smooth = total_flow_smooth_loss / num_samples
    avg_point_smooth = total_point_smooth_loss / num_samples
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Consistency Loss: {avg_consistency:.4f}")
    logger.info(f"  Flow Smooth Loss: {avg_flow_smooth:.4f}")
    logger.info(f"  Point Smooth Loss: {avg_point_smooth:.4f}")
    
    # mIoU is required
    if len(miou_list) == 0:
        raise ValueError("No valid mIoU values calculated. Please check that instance_label is provided and contains valid instances.")
    
    avg_miou = sum(miou_list) / len(miou_list)
    logger.info(f"  mIoU: {avg_miou:.4f} (from {len(miou_list)} samples)")
    
    results = {
        'consistency_loss': avg_consistency,
        'flow_smooth_loss': avg_flow_smooth,
        'point_smooth_loss': avg_point_smooth,
        'miou': avg_miou
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="src/configs/config.yaml", help="Config file")
    parser.add_argument("--data_dir", type=str, default="data/val", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    os.makedirs(config.paths.log_dir, exist_ok=True)
    _, logger = setup_logger(config.paths.log_dir, name="eval")
    logger.info("Starting evaluation...")
    
    # Create dataset
    dataset = SegValDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # Create model (Teacher-Student architecture)
    model = TeacherStudentSegmentationModel(
        use_pretrained=config.model.use_pretrained,
        pretrained_name=config.model.pretrained_name,
        feature_dim=config.model.feature_dim,
        num_instances=config.data.num_segments,
        ema_decay=config.model.ema_decay
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    results = evaluate(model, dataloader, device, config, logger)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

