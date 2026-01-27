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

