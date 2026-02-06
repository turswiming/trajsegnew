import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_util import Seq
from .pointnet2_util import (
    PointnetFPModule,
    PointnetSAModuleMSG,
    PointnetSAModule,
)
from .transformer_util import MaskFormerHead

BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}


class MaskFormer3D(nn.Module):
    """
    A 3D object segmentation network, combing PointNet++ and MaskFormer.
    """

    def __init__(
        self,
        n_slot,
        n_point=2048,
        point_feats_dim=3,
        use_xyz=True,
        bn=BN_CONFIG,
        n_transformer_layer=2,
        transformer_embed_dim=256,
        scale=1,
        transformer_input_pos_enc=False,
    ):
        super().__init__()

        # PointNet++ encoder & decoder to extract point embeddings
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=int(n_point / 4),
                radii=[1, 2],
                nsamples=[64, 64],
                mlps=[
                    [3, 32 * scale, 32 * scale, 32 * scale],
                    [3, 32 * scale, 32 * scale, 64 * scale],
                ],
                use_xyz=use_xyz,
                bn=bn,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=int(n_point / 8),
                radius=4,
                nsample=64,
                mlp=[32 * scale + 64 * scale, 64 * scale, 64 * scale, 128 * scale],
                use_xyz=use_xyz,
                bn=bn,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=int(n_point / 16),
                radius=8,
                nsample=64,
                mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale],
                use_xyz=use_xyz,
                bn=bn,
            )
        )
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(
                mlp=[64 * scale + 3, 64 * scale, 64 * scale, 64 * scale], bn=bn
            )
        )
        self.FP_modules.append(
            PointnetFPModule(
                mlp=[32 * scale + 64 * scale + 128 * scale, 64 * scale, 64 * scale],
                bn=bn,
            )
        )
        self.FP_modules.append(
            PointnetFPModule(
                mlp=[128 * scale + 256 * scale, 128 * scale, 128 * scale], bn=bn
            )
        )

        # MaskFormer head
        self.MF_head = MaskFormerHead(
            n_slot=n_slot,
            input_dim=256 * scale,
            n_transformer_layer=n_transformer_layer,
            transformer_embed_dim=transformer_embed_dim,
            transformer_n_head=8,
            transformer_hidden_dim=transformer_embed_dim,
            input_pos_enc=transformer_input_pos_enc,
        )
        self.object_mlp = (
            Seq(transformer_embed_dim)
            .conv1d(transformer_embed_dim, bn=bn)
            .conv1d(64 * scale, activation=None)
        )

    def forward(self, pc):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param point_feats: (B, N, 3) torch.Tensor.
        :return:
            mask: (B, N, K) torch.Tensor.
        """
        batch_dict = None
        if isinstance(pc, dict):
            print("begin forward with dict input")
            batch_dict = pc
            # Convert flattened batch_dict to padded batch
            coords = batch_dict['coord']
            batch_idxs = batch_dict['batch']
            batch_size = int(batch_idxs.max().item() + 1)
            print("begin padding pc")
            # Find max points
            unique, counts = torch.unique(batch_idxs, return_counts=True)
            max_points = counts.max().item()
            print(f"max points per batch: {max_points}")
            device = coords.device
            padded_pc = torch.zeros((batch_size, max_points, 3), device=device)
            # Mask to keep track of real points
            real_points_mask = torch.zeros((batch_size, max_points), dtype=torch.bool, device=device)
            
            for i in range(batch_size):
                mask_i = (batch_idxs == i)
                pts = coords[mask_i]
                n = pts.shape[0]
                padded_pc[i, :n] = pts
                real_points_mask[i, :n] = True
            
            # Replace pc with padded tensor
            pc = padded_pc
            
        point_feats = pc  # Use coordinates as initial point features
        # Extract point embeddings with PointNet++ encoder & decoder
        l_pc, l_feats = [pc], [point_feats.transpose(1, 2).contiguous()]
        for i in range(len(self.SA_modules)):
            li_pc, li_feats = self.SA_modules[i](l_pc[i], l_feats[i])
            l_pc.append(li_pc)
            l_feats.append(li_feats)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_feats[i - 1] = self.FP_modules[i](
                l_pc[i - 1], l_pc[i], l_feats[i - 1], l_feats[i]
            )

        # Extract object embeddings with MaskFormer head
        slot = self.MF_head(l_feats[-1].transpose(1, 2), l_pc[-1])  # (B, K, D)
        slot = self.object_mlp(slot.transpose(1, 2))  # (B, D, K)
        mask = (
            torch.einsum(
                "bdn,bdk->bnk", F.normalize(l_feats[0], dim=1), F.normalize(slot, dim=1)
            )
            / 0.05
        )
        
        if batch_dict is not None:
             # Reconstruct output
             valid_outputs = []
             for i in range(batch_size):
                 valid = real_points_mask[i]
                 # mask[i] is (Max_N, K)
                 valid_outputs.append(mask[i][valid]) # (Real_N, K)
             
             final_mask = torch.cat(valid_outputs, dim=0).permute(1, 0) # (Total, K) -> (K, Total)
             return {
                'pred_masks': final_mask,
                'upsampled_batch': batch_dict['batch']
             }
        
        return mask
        mask = mask.softmax(dim=-1)
        return mask


# Test the network implementation
if __name__ == "__main__":
    segnet = MaskFormer3D(
        n_slot=8,
        use_xyz=True,
        n_transformer_layer=2,
        transformer_embed_dim=128,
        transformer_input_pos_enc=False,
    ).cuda()
    pc = torch.randn(size=(4, 114514, 3)).cuda()
    point_feats = torch.randn(size=(4, 114514, 3)).cuda()
    mask = segnet(pc, point_feats)
    print(mask.shape)

    print(
        "Number of parameters:",
        sum(p.numel() for p in segnet.parameters() if p.requires_grad),
    )
    print(
        "Number of parameters in PointNet++ encoder:",
        sum(p.numel() for p in segnet.SA_modules.parameters() if p.requires_grad),
    )
    print(
        "Number of parameters in PointNet++ decoder:",
        sum(p.numel() for p in segnet.FP_modules.parameters() if p.requires_grad),
    )
    print(
        "Number of parameters in MaskFormer head:",
        sum(p.numel() for p in segnet.MF_head.parameters() if p.requires_grad),
    )
