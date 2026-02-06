import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add AccFlow to path
accflow_path = "/root/autodl-tmp/AccFlow"
if accflow_path not in sys.path:
    sys.path.insert(0, accflow_path)

from src.models.basic.sparse_encoder import MinkUNet, SparseVoxelNet
from src.models.OGCModel.transformer_util import MaskFormerHead
from src.models.OGCModel.nn_util import Seq
import spconv.pytorch as spconv

class DeltaFlowSeg(nn.Module):
    def __init__(self, 
                 voxel_size=[0.2, 0.2, 0.2],
                 point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size=[512, 512, 32],
                 planes=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16], 
                 num_layer=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 n_slot=50, # Number of queries
                 transformer_embed_dim=256,
                 n_transformer_layer=2,
                 bn={"class": "GroupNorm", "num_groups": 4} # Default BN config
                 ):
        super().__init__()
        
        # NOTE(Qingwen) [0]: point feat input channel, [-1]: voxel feat output channel
        point_output_ch = planes[0]
        voxel_output_ch = planes[-1]
        
        self.pc2voxel = SparseVoxelNet(
            voxel_size=voxel_size,
            pseudo_image_dims=grid_feature_size, 
            point_cloud_range=point_cloud_range,
            feat_channels=point_output_ch, 
            decay_factor=1.0
        )
        self.backbone = MinkUNet(planes, num_layer)
        
        self.voxel_spatial_shape = grid_feature_size
        
        # Feature dimension: Voxel Feature + Point Feature
        self.combined_feature_dim = voxel_output_ch + point_output_ch
        
        # MaskFormer Head
        self.MF_head = MaskFormerHead(
            n_slot=n_slot,
            input_dim=self.combined_feature_dim,
            n_transformer_layer=n_transformer_layer,
            transformer_embed_dim=transformer_embed_dim,
            transformer_n_head=8,
            transformer_hidden_dim=transformer_embed_dim, 
            input_pos_enc=False # Use coordinates directly if needed, but here we pass features
        )
        
        # Object MLP for projecting slots to mask space
        self.object_mlp = (
            Seq(transformer_embed_dim)
            .conv1d(transformer_embed_dim, bn=bn)
            .conv1d(self.combined_feature_dim, activation=None) # Project match feature dim
        )

    def forward(self, pc):
        """
        Args:
            pc: Input point cloud. Can be:
                - dict from OGCPointNet data preparation containing 'coord' (flattened) and 'batch'
                - or simple tensor BxNx3 (not handled here as we expect trajseg structure)
        """
        # 1. Unpack input
        batch_dict = None
        if isinstance(pc, dict):
            # OGCPointNet / TrajSeg style dictionary
            batch_dict = pc
            coords = batch_dict['coord'] # (Total_N, 3)
            batch_idxs = batch_dict['batch'] # (Total_N,)
            
            # Reconstruct list of point clouds for SparseVoxelNet
            # SparseVoxelNet expects a list of pcs for its voxelizer (implicitly handled via input_dict)
            # Actually DeltaFlow inputs a dict like input_dict['pc0s'] = [pc1, pc2, ...]
            
            batch_size = int(batch_idxs.max().item() + 1)
            pc_list = []
            for i in range(batch_size):
                mask_i = (batch_idxs == i)
                pc_list.append(coords[mask_i])
            
        else:
            # Assume list of tensors or tensor
            raise NotImplementedError("Only dictionary input (coord, batch) supported for now")

        # 2. Run Backbone (Voxelization + MinkUNet)
        # Construct input dict for DeltaFlow components
        #         
        # Voxelize
        # self.pc2voxel typically takes input_dict, calls voxelizer, processes batch
        # Let's manually call what we need to avoid the multi-frame logic in pc2voxel.forward
        
        # Use simple flow from SparseVoxelNet.voxelizer -> feature_net -> concat
        voxel_info_list = self.pc2voxel.voxelizer(pc_list)
        # process_batch returns (voxel_feats_sp, coors_batch_sp, point_feats_lst) if requested
        voxel_feats_sp, coors_batch_sp, pc0_point_feats_lst = self.pc2voxel.process_batch(
            voxel_info_list, if_return_point_feats=True
        )
        sparse_max_size = [batch_size, *self.voxel_spatial_shape, self.pc2voxel.num_feature]
        sparse_pc0 = torch.sparse_coo_tensor(coors_batch_sp.t(), voxel_feats_sp, size=sparse_max_size)
        features = sparse_pc0.coalesce().values()
        indices = sparse_pc0.coalesce().indices().t().to(dtype=torch.int32)
        all_pcdiff_sparse = spconv.SparseConvTensor(features.contiguous(), indices.contiguous(), self.voxel_spatial_shape, batch_size)

        # Backbone forward
        backbone_res = self.backbone(all_pcdiff_sparse) # Sparse Tensor Output
        
        # 3. Retrieve per-point features
        # Map voxel features back to points
        # Structure of backbone_res is Dense-like sparse tensor from spconv
        voxel_feats_dense = backbone_res.dense() # (B, C, Z, Y, X)
        
        # Gather features for each point
        point_features_list = []
        for i in range(batch_size):
            # Info for this sample
            voxel_coords = voxel_info_list[i]['voxel_coords'] # (N_i, 3) -> (x, y, z)
            # voxel_coords are int indices
            
            # Retrieve features: voxel_feats_dense[i, :, z, y, x]
            # voxel_coords is (N, 3) => columns are X, Y, Z (check SparseVoxelNet/DynamicVoxelizer)
            # usually spconv uses ZYX ordering in sparse tensor, but DynamicVoxelizer might return format
            # Let's check flow4d_module.py: voxel_feat[:, voxel_coords[:,2], voxel_coords[:,1], voxel_coords[:,0]].T
            # So coordinates are [x, y, z]. Indices are [z, y, x].
            
            # (C, N) -> (N, C)
            interp_voxel_feats = voxel_feats_dense[i, :, voxel_coords[:, 2].long(), voxel_coords[:, 1].long(), voxel_coords[:, 0].long()].T
            
            # Raw point features (from PillarFeatureNet)
            raw_point_feats = pc0_point_feats_lst[i] # (N, C_point)
            
            # Concatenate
            combined_feats = torch.cat([interp_voxel_feats, raw_point_feats], dim=-1) # (N, C_total)
            point_features_list.append(combined_feats)
            
        # 4. MaskFormer Head
        # MaskFormerHead expects (B, N, C) padded? 
        # But our input points have variable lengths.
        # Handling variable lengths with Transformer often requires padding and masks.
        # But wait, OGCPointNet handles this by padding up to max_points.
        
        # Padding
        max_points = max([p.shape[0] for p in point_features_list])
        device = coords.device
        
        padded_feats = torch.zeros((batch_size, max_points, self.combined_feature_dim), device=device)
        padded_pos = torch.zeros((batch_size, max_points, 3), device=device) # For pos enc if needed
        real_points_mask = torch.zeros((batch_size, max_points), dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            n = point_features_list[i].shape[0]
            padded_feats[i, :n] = point_features_list[i]
            padded_pos[i, :n] = voxel_info_list[i]['points']
            real_points_mask[i, :n] = True
            
        # Forward Head
        # MF_head returns `slot` (B, K, D)
        slot = self.MF_head(padded_feats, padded_pos)
        
        # Project Slots
        # slot: (B, K, D) -> (B, D, K) for conv1d
        slot_proj = self.object_mlp(slot.transpose(1, 2)) # (B, C_combined, K)
        
        # Compute Mask
        # Padded Feats: (B, N, C_combined)
        # Slot Proj: (B, C_combined, K)
        # Mask: (B, N, K)
        
        # Normalize for cosine similarity (usually good for maskformer)
        mask = torch.einsum(
            "bnc,bck->bnk", 
            F.normalize(padded_feats, dim=2), 
            F.normalize(slot_proj, dim=1)
        ) / 0.05
        
        # 5. Output Formatting (Standard TrajSeg)
        # We need to flatten back to match input 'batch_dict' order
        # Map outputs back to original point cloud size (including invalid points)
        num_slots = mask.shape[2]  # K
        device = coords.device
        
        # For each batch, create output matching original point cloud size
        full_outputs = []
        for i in range(batch_size):
            n_valid = point_features_list[i].shape[0]  # Number of valid points after voxelization
            n_original = pc_list[i].shape[0]  # Original number of points
            
            # Valid mask for this batch: (N_valid, K)
            valid_mask = mask[i][real_points_mask[i]]  # (N_valid, K)
            
            # Create full output tensor matching original point cloud size
            full_mask = torch.zeros((n_original, num_slots), device=device)
            
            if n_valid > 0:
                # Use point_idxes from voxelizer for direct mapping
                # DynamicVoxelizer returns point_idxes which records which original points were kept
                point_idxes = voxel_info_list[i]['point_idxes']  # (N_valid,) indices into original points
                full_mask[point_idxes] = valid_mask
            
            full_outputs.append(full_mask)
        
        final_mask = torch.cat(full_outputs, dim=0).permute(1, 0) # (Total, K) -> (K, Total)
        #apply softmax to final_mask
        final_mask = F.softmax(final_mask, dim=0)
        return {
            'pred_masks': final_mask,
            'upsampled_batch': batch_idxs
        }

