"""
Point Smoothness Loss implementation for point cloud segmentation.

This module implements a comprehensive smoothness loss for point cloud segmentation,
combining k-nearest neighbors (KNN) and ball query approaches to encourage spatially
coherent segmentation masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather, ball_query


class KnnLoss(nn.Module):
    """
    K-nearest neighbors based smoothness loss component.
    """

    def __init__(self, k=8, radius=0.1, cross_entropy=False, loss_norm=1, **kwargs):
        """
        Initialize the KNN Loss component.

        Args:
            k: Number of nearest neighbors
            radius: Maximum neighbor distance
            cross_entropy: Use cross-entropy loss instead of norm
            loss_norm: Which norm to use (1 or 2)
        """
        super().__init__()
        self.k = k
        self.radius = radius
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def forward(self, pc, mask):
        """
        Compute the KNN-based smoothness loss.

        Args:
            pc: Point cloud coordinates [B, N, 3]
            mask: Segmentation mask [B, N, K]

        Returns:
            Computed KNN smoothness loss
        """
        mask = mask.permute(0, 2, 1).contiguous()  # (B, Kclasses, N)
        dists, idx, _ = knn_points(pc, pc, K=self.k, return_nn=True)  # dists: (B, N, K)
        euclidean_dists = dists.sqrt()
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        over_radius = euclidean_dists > self.radius
        idx = idx.clone()
        idx[over_radius] = tmp_idx[over_radius]
        idx = idx.to(dtype=torch.int64)
        num_points = mask.shape[2]
        idx = idx.clamp_(min=0, max=num_points - 1)
        feats = mask.permute(0, 2, 1).contiguous()
        nn_mask = knn_gather(feats, idx.detach()).permute(0, 3, 1, 2).contiguous()
        
        if self.cross_entropy:
            mask = mask.unsqueeze(3).repeat(1, 1, 1, self.k).detach()
            loss = F.binary_cross_entropy(nn_mask, mask, reduction='none').sum(dim=1).mean(dim=-1)
        else:
            ref = mask.unsqueeze(3)
            loss = (ref - nn_mask).norm(p=self.loss_norm, dim=1).mean(dim=-1)
        return loss.mean()


class BallQLoss(nn.Module):
    """
    Ball query based smoothness loss component.
    """

    def __init__(self, k=16, radius=0.2, cross_entropy=False, loss_norm=1, **kwargs):
        """
        Initialize the Ball Query Loss component.

        Args:
            k: Maximum points in ball
            radius: Ball radius
            cross_entropy: Use cross-entropy loss instead of norm
            loss_norm: Which norm to use (1 or 2)
        """
        super().__init__()
        self.k = k
        self.radius = radius
        self.cross_entropy = cross_entropy
        self.loss_norm = loss_norm

    def forward(self, pc, mask):
        """
        Compute the ball query based smoothness loss.

        Args:
            pc: Point cloud coordinates [B, N, 3]
            mask: Segmentation mask [B, N, K]

        Returns:
            Computed ball query smoothness loss
        """
        mask = mask.permute(0, 2, 1).contiguous()  # (B, Kclasses, N)
        idx = ball_query(pc, pc, K=self.k, radius=self.radius)
        
        if not torch.is_tensor(idx):
            idx = getattr(idx, 'idx', idx)
        
        B, N, K = idx.shape
        device = idx.device
        self_idx = torch.arange(N, device=device).view(1, N, 1).repeat(B, 1, K)
        idx = torch.where(idx < 0, self_idx, idx)
        idx = idx.to(dtype=torch.int64)
        num_points = mask.shape[2]
        idx = idx.clamp_(min=0, max=num_points - 1)
        feats = mask.permute(0, 2, 1).contiguous()
        nn_mask = knn_gather(feats, idx.detach()).permute(0, 3, 1, 2).contiguous()
        
        if self.cross_entropy:
            mask = mask.unsqueeze(3).repeat(1, 1, 1, self.k).detach()
            loss = F.binary_cross_entropy(nn_mask, mask, reduction='none').sum(dim=1).mean(dim=-1)
        else:
            ref = mask.unsqueeze(3)
            loss = (ref - nn_mask).norm(p=self.loss_norm, dim=1).mean(dim=-1)
        return loss.mean()


class PointSmoothLoss(nn.Module):
    """
    Combined point smoothness loss for segmentation.
    """

    def __init__(self, w_knn=3, w_ball_q=1, knn_loss_params=None, ball_q_loss_params=None):
        """
        Initialize the combined Point Smoothness Loss.

        Args:
            w_knn: Weight for KNN loss
            w_ball_q: Weight for ball query loss
            knn_loss_params: Parameters for KNN loss
            ball_q_loss_params: Parameters for ball query loss
        """
        super().__init__()
        if knn_loss_params is None:
            knn_loss_params = {}
        if ball_q_loss_params is None:
            ball_q_loss_params = {}
        
        self.knn_loss = KnnLoss(**knn_loss_params)
        self.ball_q_loss = BallQLoss(**ball_q_loss_params)
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc, mask):
        """
        Compute the combined smoothness loss.

        Args:
            pc: List of point cloud coordinates, each of shape [N, 3]
            mask: List of segmentation masks, each of shape [K, N]

        Returns:
            Combined smoothness loss averaged across the batch
        """
        batch_size = len(pc)
        mask_reshaped = [item.permute(1, 0).unsqueeze(0) for item in mask]
        pc = [item.unsqueeze(0) for item in pc]
        loss = torch.zeros(batch_size, device=pc[0].device)
        
        for i in range(batch_size):
            loss[i] = (
                self.w_knn * self.knn_loss(pc[i], mask_reshaped[i]) +
                self.w_ball_q * self.ball_q_loss(pc[i], mask_reshaped[i])
            )
        
        return loss.mean()

