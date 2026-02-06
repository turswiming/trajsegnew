"""
Evaluation metrics for scene flow.
"""

import torch
import torch.nn.functional as F


def compute_epe(pred_flow, gt_flow, mask=None):
    """
    Compute End-Point Error (EPE).

    Args:
        pred_flow: [N, 3] predicted flow
        gt_flow: [N, 3] ground truth flow
        mask: [N] optional mask for valid points

    Returns:
        epe: Average EPE
    """
    error = pred_flow - gt_flow
    epe = torch.norm(error, dim=1)
    
    if mask is not None:
        epe = epe[mask]
    
    return epe.mean().item()


def compute_acc_strict(pred_flow, gt_flow, mask=None, threshold=0.05):
    """
    Compute strict accuracy (EPE < threshold).

    Args:
        pred_flow: [N, 3] predicted flow
        gt_flow: [N, 3] ground truth flow
        mask: [N] optional mask for valid points
        threshold: EPE threshold

    Returns:
        acc: Accuracy percentage
    """
    error = pred_flow - gt_flow
    epe = torch.norm(error, dim=1)
    
    if mask is not None:
        epe = epe[mask]
    
    acc = (epe < threshold).float().mean().item() * 100.0
    return acc


def compute_acc_relax(pred_flow, gt_flow, mask=None, threshold=0.1):
    """
    Compute relaxed accuracy (EPE < threshold).

    Args:
        pred_flow: [N, 3] predicted flow
        gt_flow: [N, 3] ground truth flow
        mask: [N] optional mask for valid points
        threshold: EPE threshold

    Returns:
        acc: Accuracy percentage
    """
    return compute_acc_strict(pred_flow, gt_flow, mask, threshold)


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
    if pred_mask.shape[1] != gt_mask.shape[1]:
        min_n = min(pred_mask.shape[1], gt_mask.shape[1])
        pred_mask = pred_mask[:, :min_n]
        gt_mask = gt_mask[:, :min_n]
    
    pred_mask = pred_mask.contiguous()
    if pred_mask.dim() == 2:
        pred_mask = torch.softmax(pred_mask, dim=0)
        pred_mask_argmax = torch.argmax(pred_mask, dim=0)
        pred_mask = F.one_hot(pred_mask_argmax, num_classes=pred_mask.shape[0]).permute(1, 0).float()
    
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5
    
    gt_mask_size = torch.sum(gt_mask, dim=1)
    pred_mask_size = torch.sum(pred_mask, dim=1)
    
    max_iou_list = []
    for j in range(gt_mask.shape[0]):
        if gt_mask_size[j] < min_points:
            continue
        
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

