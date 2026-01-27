"""
Invariance Loss for segmentation consistency across views.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


def match_mask_by_iou(mask1, mask2):
    """
    Match individual objects in two object masks by Hungarian algorithm.
    
    Args:
        mask1: (B, N, K) torch.Tensor
        mask2: (B, N, K) torch.Tensor
    
    Returns:
        perm: (B, K, K) torch.Tensor, permutation for alignment
    """
    n_batch, _, n_object = mask1.size()
    segm_pred1 = mask1.argmax(-1).detach()
    segm_pred2 = mask2.argmax(-1).detach()
    segm_pred1 = torch.eye(n_object, dtype=torch.float32, device=segm_pred1.device)[segm_pred1]
    segm_pred2 = torch.eye(n_object, dtype=torch.float32, device=segm_pred2.device)[segm_pred2]

    # Match according to IoU
    intersection = torch.einsum('bng,bnp->bgp', segm_pred1, segm_pred2)  # (B, K, K)
    union = torch.sum(segm_pred1, dim=1).unsqueeze(-1) + torch.sum(segm_pred2, dim=1, keepdim=True) - intersection
    iou = intersection / union.clamp(1e-10)
    perm = []
    for b in range(n_batch):
        iou_score = iou[b].cpu().numpy()
        _, col_ind = linear_sum_assignment(iou_score, maximize=True)
        perm.append(col_ind)
    perm = torch.from_numpy(np.stack(perm, 0))
    perm = torch.eye(n_object, dtype=torch.float32, device=segm_pred1.device)[perm]
    return perm


class InvarianceLoss(nn.Module):
    """
    Minimize the difference between matched per-point segmentation.
    """
    
    def __init__(self, cross_entropy=False, loss_norm=2):
        """
        Initialize Invariance Loss.
        
        Args:
            cross_entropy: Whether to use cross-entropy loss
            loss_norm: Norm for distance calculation (1 or 2)
        """
        super().__init__()
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def distance(self, mask1, mask2):
        """
        Compute distance between two masks.
        
        Args:
            mask1: (B, N, K) torch.Tensor, prediction
            mask2: (B, N, K) torch.Tensor, target
        
        Returns:
            loss: scalar torch.Tensor
        """
        if self.cross_entropy:
            loss = F.binary_cross_entropy(mask1, mask2, reduction='none').sum(dim=1)
        else:
            loss = (mask1 - mask2).norm(p=self.loss_norm, dim=-1)
        return loss.mean()

    def forward(self, mask1, mask2):
        """
        Compute invariance loss between two masks.
        
        Args:
            mask1: (B, N, K) torch.Tensor
            mask2: (B, N, K) torch.Tensor
        
        Returns:
            loss: scalar torch.Tensor
        """
        # Align the object ordering in two views
        perm2 = match_mask_by_iou(mask1, mask2)
        target_mask1 = torch.einsum('bij,bnj->bni', perm2, mask2).detach()
        perm1 = match_mask_by_iou(mask2, mask1)
        target_mask2 = torch.einsum('bij,bnj->bni', perm1, mask1).detach()

        # Enforce the invariance
        loss = self.distance(mask1, target_mask1) + self.distance(mask2, target_mask2)
        return loss

