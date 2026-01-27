"""
Trajectory Loss 3D implementation for trajectory-based segmentation learning.
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class BinaryMask(torch.autograd.Function):
    """
    Custom autograd function for binary mask using Straight-Through Estimator.
    Reference: FlowSmoothLoss
    """
    @staticmethod
    def forward(ctx, mask):
        # mask shape: [K, N]
        # argmax along dim=0 to get which mask is active for each point
        argmax_index = torch.argmax(mask, dim=0)  # [N] or scalar if N=1
        one_hot = F.one_hot(argmax_index, num_classes=mask.shape[0])
        
        # Transpose to [K, N] and convert to float
        binary = one_hot.float().permute(1, 0)  # [K, N]
        return binary
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TrajectoryLoss_3d:
    def __init__(self, cfg, r=2, criterion='L2',device=None, downsample_ratio=1):
        self.device = device
        self.cfg = cfg
        self.r = r
        self.downsample_ratio = downsample_ratio
        if isinstance(criterion, str):
            self.criterion = nn.MSELoss(reduction="sum") if criterion.upper() == 'L2' else nn.L1Loss(reduction="sum")
        else:
            self.criterion = criterion

    def __call__(self, sample, flow, mask_softmaxed, it, train=True):
        return self.loss(sample, flow, mask_softmaxed, it, train=train)
    
    def pi_func(self, mask_all, point_position, traj_single_frame):
        """
        Convert mask to binary using argmax (similar to FlowSmoothLoss).
        
        :param mask_all: (K, N) all masks for all segments
        :param point_position: (N, 3) point cloud positions (not used, kept for interface compatibility)
        :param traj_single_frame: (3, num_tracks) trajectory positions (not used, kept for interface compatibility)
        :return: (K, N) binary mask values
        """
        # Use BinaryMask to get binary mask via argmax
        # mask_all: (K, N)
        binary_mask = BinaryMask.apply(mask_all)  # (K, N)
        
        return binary_mask


    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        """
        Compute trajectory loss.
        
        Args:
            sample: Dictionary or list of dictionaries with:
                - trajectories: (frame_length, N, 3) trajectory data (each point has frame_length frames)
                - point_cloud: (N, 3) point cloud positions (first frame)
                - abs_index: absolute frame index (default: 0)
            flow: List of flow tensors or single tensor (not used in this loss)
            mask_softmaxed: (B, K, N) mask logits/probabilities
            it: iteration number (not used)
            train: whether in training mode (if False, skip computation)
        
        Returns:
            Total trajectory loss
        """
        if not train:
            # Skip trajectory loss during validation/evaluation
            return torch.tensor(0.0, device=self.device)
        
        # Handle batch input: sample can be a list or single dict
        if isinstance(sample, list):
            samples = sample
        else:
            samples = [sample]
        
        # Handle mask_softmaxed: should be (B, K, N)
        if mask_softmaxed.dim() == 4:
            # (B, K, H, W) -> (B, K, H*W)
            B, K, H, W = mask_softmaxed.shape
            mask_softmaxed = mask_softmaxed.view(B, K, H * W)
        elif mask_softmaxed.dim() == 3:
            B, K, N = mask_softmaxed.shape
        else:
            raise ValueError(f"Unexpected mask_softmaxed shape: {mask_softmaxed.shape}")
        
        total_loss = 0.0
        
        for b in range(B):
            if b >= len(samples):
                continue
                
            sample_b = samples[b]
            
            # Check if trajectory data exists
            if 'trajectories' not in sample_b:
                continue
            
            trajectories = sample_b['trajectories']  # (frame_length, N, 3)

            point_position = trajectories[0]  # (N, 3)

            point_position = point_position.to(self.device)
            abs_index = sample_b.get('abs_index', 0)
            
            # trajectories.shape (frame_length, N, 3)
            # Convert from numpy to tensor if needed
            if isinstance(trajectories, np.ndarray):
                trajectories = torch.from_numpy(trajectories).float()
            trajectories = trajectories.to(self.device)
            
            frame_length, num_tracks, _ = trajectories.shape
            
            # Pt: current frame trajectory positions (3, num_tracks)
            Pt = trajectories[abs_index].permute(1, 0)  # Pt [3, num_tracks]
            mask_all = mask_softmaxed[b]  # (K, N)

            #dowmsample
            Pt = Pt[::self.downsample_ratio]
            mask_all = mask_all[:, ::self.downsample_ratio]
            point_position = point_position[::self.downsample_ratio]
            trajectories = trajectories[:,::self.downsample_ratio]
            # Convert from [frame_length, num_tracks, 3] to [frame_length * 3, num_tracks]
            # P: all frames trajectory data
            traj_3d = trajectories.permute(2, 0, 1).reshape(-1, trajectories.shape[1])  # (frame_length * 3, num_tracks)
            # Get binary mask using argmax: (K, N)
            mask_binary = self.pi_func(mask_all, point_position, Pt)  # (K, N)
            
            for k in range(K):
                # Get binary mask for this segment: (N,)
                Mk_hat = mask_binary[k]  # (N,) binary mask for segment k
                
                # Pk: weighted trajectory by binary mask
                # traj_3d: (frame_length * 3, N)
                # Mk_hat: (N,)
                # Multiply each column (trajectory) by corresponding mask value
                Pk = traj_3d * Mk_hat.unsqueeze(0)  # Pk [frame_length * 3, N]
                
                if Pk.shape[1] == 0 or Mk_hat.sum() == 0:
                    # Skip if no points in this mask
                    continue
                
                try:
                    # Do SVD
                    U, S, Vh = torch.linalg.svd(Pk, full_matrices=False)
                except RuntimeError:
                    print('RuntimeError in SVD')
                    S = torch.zeros(min(Pk.shape), device=self.device)
                    seg_loss = torch.sum(S[self.r:])
                else:
                    # Calculate loss
                    seg_loss = torch.mean(S[self.r:])
                
                total_loss += seg_loss
        
        return total_loss
