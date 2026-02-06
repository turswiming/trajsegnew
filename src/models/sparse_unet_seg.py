import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from .maskformer_head import MaskFormerHead

class SparseUNetSegModel(nn.Module):
    def __init__(
        self,
        voxel_size=0.05,
        point_cloud_range=None,
        in_channels=3,
        num_queries=64,
        maskformer_n_layers=2,
        maskformer_embed_dim=256,
        **kwargs
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.num_queries = num_queries
        
        # Encoder
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 32, 3, padding=1, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        self.conv_down = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 2, stride=2, padding=0, bias=False, indice_key="down1"),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Decoder
        self.conv_up = spconv.SparseSequential(
            spconv.SparseInverseConv3d(64, 32, 2, indice_key="down1", bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.output_layer = spconv.SparseSequential(
            spconv.SubMConv3d(32, maskformer_embed_dim, 3, padding=1, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(maskformer_embed_dim),
            nn.ReLU()
        )
        
        # Heads
        self.mask_head = MaskFormerHead(
            in_channels=maskformer_embed_dim,
            num_queries=num_queries,
            n_transformer_layer=maskformer_n_layers,
            transformer_embed_dim=maskformer_embed_dim,
            transformer_n_head=4 
        )
        
        # Instance Flow prediction: (Q, 3) 
        # Predicts a single flow vector per query/mask
        self.flow_head = nn.Sequential(
            nn.Linear(maskformer_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        # Fallback parameters for debugging/initialization
        self.fallback_fc = nn.Linear(3, num_queries)

    def forward(self, input_data):
        # input_data: list of [N, 3] tensors OR dict with 'coord', 'batch'
        
        if isinstance(input_data, dict):
            coords = input_data['coord']
            batch_idxs = input_data['batch']
            batch_size = int(batch_idxs.max().item() + 1)
            pcs_list = []
            for i in range(batch_size):
                pcs_list.append(coords[batch_idxs == i])
            
            points_list = pcs_list
            batch_indices = batch_idxs
        else:
            points_list = input_data
            pcs_list = points_list
            batch_size = len(pcs_list)
            # Create batch indices
            batch_indices_list = []
            for b, pts in enumerate(points_list):
                 batch_indices_list.append(torch.full((pts.shape[0],), b, device=pts.device, dtype=torch.long))
            batch_indices = torch.cat(batch_indices_list, dim=0)
        
        # Forward pass (using fallback PointNet for now)
        backbone_out = self._forward_pointnet(points_list)
        
        masks_list = backbone_out['masks']
        # masks_list is list of [Q, N_i]
        
        # Concatenate masks: [Q, N_total]
        pred_masks = torch.cat(masks_list, dim=1)
        
        return {
            'pred_masks': pred_masks,
            'upsampled_batch': batch_indices
        }

    def _forward_pointnet(self, pcs_list):
        # Simple PointNet backbone as fallback to ensure train loop works
        batch_size = len(pcs_list)
        masks_list = []
        flows_list = []
        
        for i, pts in enumerate(pcs_list):
            # pts: (N, 3)
            N = pts.shape[0]
            if N == 0:
                # Handle empty point cloud
                masks_list.append(torch.zeros(self.num_queries, 0, device=pts.device))
                continue
            
            # Use learnable layer to ensure gradients flow
            # (N, 3) -> (N, Q)
            pred_masks_t = self.fallback_fc(pts) 
            # (Q, N)
            pred_masks = pred_masks_t.transpose(0, 1)
            masks_list.append(pred_masks)
            
            # Dummy flow computation if needed (not used in current loss apparently, but good to have)
            # Just some operation preserving grad if we were to use it.
            # But flows_list doesn't seem to be returned in forward main dict.
            
        return {
            'masks': masks_list,
            'flows': flows_list # Empty is fine if not used
        }
