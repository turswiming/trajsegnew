"""
OGC PointNet++ model wrapper for trajectories_for_seg.
Adapts the MaskFormer3D model from OGC to work with the current training framework.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add OGC path to sys.path to import OGC modules
ogc_path = os.path.join(os.path.dirname(__file__), '../../..', 'OGC')
if ogc_path not in sys.path:
    sys.path.insert(0, ogc_path)

from models.segnet_kitti import MaskFormer3D
from utils.nn_util import Seq

BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}


class OGCPointNetModel(nn.Module):
    """
    Wrapper for OGC's MaskFormer3D (PointNet++ based) model.
    Adapts the model to work with the current training framework's input/output format.
    """
    
    def __init__(
        self,
        num_queries=32,
        n_point=81920,
        n_transformer_layer=2,
        transformer_embed_dim=256,
        transformer_input_pos_enc=False,
    ):
        """
        Initialize OGC PointNet++ model.
        
        Args:
            num_queries: Number of mask queries (K)
            n_point: Number of points (default 2048, will be adjusted based on input)
            n_transformer_layer: Number of transformer layers
            transformer_embed_dim: Transformer embedding dimension
            transformer_input_pos_enc: Whether to use input position encoding
        """
        super().__init__()
        
        self.num_queries = num_queries
        
        # Create the OGC MaskFormer3D model
        self.model = MaskFormer3D(
            n_slot=num_queries,
            n_point=n_point,
            use_xyz=True,
            bn=BN_CONFIG,
            n_transformer_layer=n_transformer_layer,
            transformer_embed_dim=transformer_embed_dim,
            transformer_input_pos_enc=transformer_input_pos_enc
        )
    
    def forward(self, point_dict, return_teacher=False, upsample_masks=False, use_batch_parallel=True):
        """
        Forward pass through the model.
        
        Args:
            point_dict: Dictionary with point cloud data
                - coord: [N, 3] point coordinates
                - feat: [N, D] point features (usually same as coord)
                - batch: [N] batch indices
            return_teacher: Not used for this model (no teacher-student architecture)
            upsample_masks: Not used (masks are already at original scale)
            use_batch_parallel: If True, use nan padding for batch parallel processing
        
        Returns:
            Dictionary with:
            - upsampled_student_masks: [K, N] masks at original scale
            - upsampled_batch: [N] batch indices
        """
        # Extract data from point_dict
        coords = point_dict['coord']  # [N, 3]
        feats = point_dict.get('feat', coords)  # [N, D] or [N, 3]
        batch = point_dict['batch']  # [N]
        
        # Get unique batch indices
        unique_batches = torch.unique(batch)
        batch_size = len(unique_batches)
        
        if use_batch_parallel and batch_size > 1:
            # 使用nan填充实现batch并行
            return self._forward_batch_parallel(coords, feats, batch, unique_batches)
        else:
            # 原来的逐个batch处理方式
            return self._forward_sequential(coords, feats, batch, unique_batches)
    
    def _forward_sequential(self, coords, feats, batch, unique_batches):
        """逐个batch处理（原始方式）"""
        all_masks = []
        all_batch_indices = []
        
        for i, batch_idx in enumerate(unique_batches):
            # Get points for this batch
            batch_mask = (batch == batch_idx)
            batch_coords = coords[batch_mask]  # [N_i, 3]
            batch_feats = feats[batch_mask]  # [N_i, D]
            
            # Reshape for OGC model: (B, N, 3) and (B, N, 3)
            batch_coords = batch_coords.unsqueeze(0)  # [1, N_i, 3]
            batch_feats = batch_feats.unsqueeze(0)  # [1, N_i, 3]
            
            # Forward through OGC model
            # Output: [B, N, K] masks
            masks = self.model(batch_coords, batch_feats)  # [1, N_i, K]
            
            # Reshape to [K, N_i]
            masks = masks.squeeze(0).transpose(0, 1)  # [K, N_i]
            
            all_masks.append(masks)
            all_batch_indices.append(batch[batch_mask])
        
        # Concatenate all batches
        upsampled_student_masks = torch.cat(all_masks, dim=1)  # [K, N_total]
        upsampled_batch = torch.cat(all_batch_indices, dim=0)  # [N_total]
        
        return {
            'upsampled_student_masks': upsampled_student_masks,
            'upsampled_batch': upsampled_batch,
        }
    
    def _forward_batch_parallel(self, coords, feats, batch, unique_batches):
        """
        使用nan填充实现batch并行处理。
        策略：将padding点放在远离真实点的位置，避免FPS选择到它们。
        """
        device = coords.device
        batch_size = len(unique_batches)
        
        # 获取每个batch的点数和数据
        batch_counts = []
        batch_masks = []
        batch_coords_list = []
        batch_feats_list = []
        
        for batch_idx in unique_batches:
            batch_mask = (batch == batch_idx)
            n_points = batch_mask.sum().item()
            batch_counts.append(n_points)
            batch_masks.append(batch_mask)
            batch_coords_list.append(coords[batch_mask])
            batch_feats_list.append(feats[batch_mask])
        
        # 找到最大点数
        max_n_points = max(batch_counts)
        
        # 计算一个远离所有真实点的位置（用于padding）
        # 使用所有点的边界框，然后向外扩展
        all_coords_min = coords.min(dim=0)[0]
        all_coords_max = coords.max(dim=0)[0]
        coord_range = all_coords_max - all_coords_min
        # padding点放在边界框外很远的地方
        padding_coord = all_coords_min - coord_range * 10.0
        
        # 创建padding后的batch tensor [B, max_N, 3]
        padded_coords = torch.zeros(
            (batch_size, max_n_points, 3), 
            device=device, 
            dtype=coords.dtype
        )
        padded_feats = torch.zeros(
            (batch_size, max_n_points, feats.shape[1]), 
            device=device, 
            dtype=feats.dtype
        )
        
        # 填充每个batch的数据
        valid_masks = []  # 记录每个batch的有效点mask
        for i in range(batch_size):
            n_points = batch_counts[i]
            
            # 填充真实数据
            padded_coords[i, :n_points] = batch_coords_list[i]
            padded_feats[i, :n_points] = batch_feats_list[i]
            
            # 填充padding点（放在远离真实点的位置）
            if n_points < max_n_points:
                padded_coords[i, n_points:] = padding_coord.unsqueeze(0).expand(
                    max_n_points - n_points, -1
                )
                padded_feats[i, n_points:] = 0.0
            
            # 创建有效点mask [max_N]
            valid_mask = torch.zeros(max_n_points, dtype=torch.bool, device=device)
            valid_mask[:n_points] = True
            valid_masks.append(valid_mask)
        
        # Forward through OGC model
        # Output: [B, max_N, K] masks
        masks = self.model(padded_coords, padded_feats)  # [B, max_N, K]
        
        # 将mask转换为 [K, N_total] 格式，只保留有效点
        all_masks = []
        all_batch_indices = []
        
        for i, batch_idx in enumerate(unique_batches):
            batch_mask = batch_masks[i]
            valid_mask = valid_masks[i]
            
            # 提取有效点的mask [K, n_points]
            batch_masks_tensor = masks[i, valid_mask].transpose(0, 1)  # [K, n_points]
            
            all_masks.append(batch_masks_tensor)
            all_batch_indices.append(batch[batch_mask])
        
        # Concatenate all batches
        upsampled_student_masks = torch.cat(all_masks, dim=1)  # [K, N_total]
        upsampled_batch = torch.cat(all_batch_indices, dim=0)  # [N_total]
        
        return {
            'upsampled_student_masks': upsampled_student_masks,
            'upsampled_batch': upsampled_batch,
        }
    
    def update_teacher(self):
        """
        Placeholder for teacher update (not used for this model).
        """
        pass

