"""
Sparse 3D Convolution UNet for Point Cloud Segmentation with MaskFormer Head.
流程：
1. 点云voxelization，保存映射关系
2. 稀疏3D卷积UNet（下采样+上采样）
3. MaskFormer分割头
4. 使用映射关系恢复点云
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import spconv as spconv_core

# 允许TF32以加速计算
spconv_core.constants.SPCONV_ALLOW_TF32 = True

from .maskformer_head import MaskFormerHead


class PointVoxelizer(nn.Module):
    """
    点云voxelization模块，保存点云到voxel的映射关系。
    """
    
    def __init__(self, voxel_size=0.05, point_cloud_range=None):
        """
        Args:
            voxel_size: voxel大小，可以是float或list[float]
            point_cloud_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        super().__init__()
        if isinstance(voxel_size, (int, float)):
            self.voxel_size = [voxel_size, voxel_size, voxel_size]
        else:
            self.voxel_size = voxel_size
        
        self.point_cloud_range = point_cloud_range
    
    def forward(self, points, batch_indices=None):
        """
        将点云voxelize，保存映射关系。
        
        Args:
            points: [N, 3] 点云坐标
            batch_indices: [N] batch索引（可选）
        
        Returns:
            voxel_coords: [M, 4] voxel坐标 [batch_idx, z, y, x]
            voxel_features: [M, C] voxel特征（初始为voxel中心坐标）
            point_to_voxel_map: [N] 每个点对应的voxel索引，-1表示无效点
            voxel_to_point_map: List[List[int]] 每个voxel包含的点索引列表
        """
        device = points.device
        N = points.shape[0]
        
        # 如果没有提供batch_indices，假设batch_size=1
        if batch_indices is None:
            batch_indices = torch.zeros(N, dtype=torch.long, device=device)
        
        # 计算点云范围
        if self.point_cloud_range is None:
            point_min = points.min(dim=0)[0] - torch.tensor(self.voxel_size, device=device) * 0.5
            point_max = points.max(dim=0)[0] + torch.tensor(self.voxel_size, device=device) * 0.5
        else:
            point_min = torch.tensor(self.point_cloud_range[:3], device=device)
            point_max = torch.tensor(self.point_cloud_range[3:], device=device)
        
        voxel_size = torch.tensor(self.voxel_size, device=device, dtype=points.dtype)
        
        # 计算voxel坐标
        voxel_coords_float = (points - point_min) / voxel_size.unsqueeze(0)
        voxel_coords = torch.floor(voxel_coords_float).long()
        
        # 过滤超出范围的点
        valid_mask = (voxel_coords >= 0).all(dim=1) & (
            voxel_coords < torch.ceil((point_max - point_min) / voxel_size).long()
        ).all(dim=1)
        
        if not valid_mask.any():
            # 如果没有有效点，返回空结果
            empty_coords = torch.zeros(0, 4, dtype=torch.long, device=device)
            empty_features = torch.zeros(0, 3, dtype=points.dtype, device=device)
            empty_map = torch.full((N,), -1, dtype=torch.long, device=device)
            return empty_coords, empty_features, empty_map, []
        
        valid_points = points[valid_mask]
        valid_voxel_coords = voxel_coords[valid_mask]
        valid_batch_indices = batch_indices[valid_mask]
        valid_point_indices = torch.arange(N, device=device)[valid_mask]
        
        # 创建voxel坐标（包含batch索引）
        voxel_coords_with_batch = torch.cat([
            valid_batch_indices.unsqueeze(1),
            valid_voxel_coords[:, [2, 1, 0]]  # 转换为 [batch, z, y, x] 格式
        ], dim=1)
        
        # 找到唯一的voxel
        # 使用hash来快速找到唯一voxel
        voxel_max = valid_voxel_coords.max(dim=0)[0]
        max_yz = (voxel_max[1] + 1) * (voxel_max[2] + 1)
        max_z = voxel_max[2] + 1
        
        voxel_hash = (
            valid_voxel_coords[:, 0] * max_yz +
            valid_voxel_coords[:, 1] * max_z +
            valid_voxel_coords[:, 2]
        )
        
        # 为每个batch分别处理
        unique_batches = torch.unique(valid_batch_indices)
        all_voxel_coords = []
        all_voxel_features = []
        all_point_to_voxel_map = torch.full((N,), -1, dtype=torch.long, device=device)
        all_voxel_to_point_map = []
        
        voxel_idx_offset = 0
        
        for batch_idx in unique_batches:
            batch_mask = (valid_batch_indices == batch_idx)
            batch_voxel_hash = voxel_hash[batch_mask]
            batch_point_indices = valid_point_indices[batch_mask]
            batch_voxel_coords_with_batch = voxel_coords_with_batch[batch_mask]
            batch_points = valid_points[batch_mask]
            
            # 找到唯一voxel
            unique_voxels, inverse_indices = torch.unique(
                batch_voxel_hash, return_inverse=True
            )
            
            n_voxels = len(unique_voxels)
            
            # 计算每个voxel的特征（使用voxel中心坐标）
            unique_voxel_coords = torch.zeros(n_voxels, 3, dtype=torch.long, device=device)
            voxel_centers = torch.zeros(n_voxels, 3, dtype=points.dtype, device=device)
            
            for i in range(n_voxels):
                # 找到第一个属于这个voxel的点
                first_idx = (inverse_indices == i).nonzero(as_tuple=True)[0][0]
                unique_voxel_coords[i] = valid_voxel_coords[batch_mask][first_idx]
                # 计算voxel中心
                voxel_centers[i] = point_min + (unique_voxel_coords[i].float() + 0.5) * voxel_size
            
            # 构建映射关系
            for i, point_idx in enumerate(batch_point_indices):
                voxel_idx = inverse_indices[i] + voxel_idx_offset
                all_point_to_voxel_map[point_idx] = voxel_idx
            
            # 构建voxel到点的映射
            for i in range(n_voxels):
                point_indices_in_voxel = batch_point_indices[inverse_indices == i].tolist()
                all_voxel_to_point_map.append(point_indices_in_voxel)
            
            # 获取唯一的voxel坐标（包含batch）
            unique_voxel_coords_with_batch = torch.zeros(n_voxels, 4, dtype=torch.long, device=device)
            for i in range(n_voxels):
                first_idx = (inverse_indices == i).nonzero(as_tuple=True)[0][0]
                unique_voxel_coords_with_batch[i] = batch_voxel_coords_with_batch[first_idx]
            
            all_voxel_coords.append(unique_voxel_coords_with_batch)
            all_voxel_features.append(voxel_centers)
            
            voxel_idx_offset += n_voxels
        
        # 合并所有batch的结果
        voxel_coords = torch.cat(all_voxel_coords, dim=0)  # [M, 4]
        voxel_features = torch.cat(all_voxel_features, dim=0)  # [M, 3]
        
        return voxel_coords, voxel_features, all_point_to_voxel_map, all_voxel_to_point_map


class SparseResidualBlock(nn.Module):
    """稀疏3D卷积残差块"""
    
    def __init__(self, in_channels, out_channels, indice_key, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if stride == 1 and in_channels == out_channels:
            self.conv = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, 
                                 stride=1, padding=1, bias=False, indice_key=indice_key,
                                 algo=spconv.ConvAlgo.Native),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(out_channels, out_channels, kernel_size=3,
                                 stride=1, padding=1, bias=False, indice_key=indice_key,
                                 algo=spconv.ConvAlgo.Native),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.conv = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False, indice_key=indice_key,
                                   algo=spconv.ConvAlgo.Native),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(out_channels, out_channels, kernel_size=3,
                                 stride=1, padding=1, bias=False, indice_key=indice_key,
                                 algo=spconv.ConvAlgo.Native),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
        # 如果输入输出通道数不同，需要投影
        if in_channels != out_channels or stride != 1:
            self.shortcut = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size=1,
                                   stride=stride, bias=False, indice_key=indice_key + '_shortcut',
                                   algo=spconv.ConvAlgo.Native),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = None
    
    def forward(self, x):
        identity = x
        out = self.conv(x)
        
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        out = out.replace_feature(self.relu(out.features + identity.features))
        return out


class SparseUNet(nn.Module):
    """
    稀疏3D卷积UNet，包含下采样和上采样。
    """
    
    def __init__(
        self,
        in_channels=3,
        base_channels=16,
        channels=[16, 32, 64, 128, 256],
        num_blocks=[2, 2, 2, 2, 2],
        sparse_shape=None
    ):
        """
        Args:
            in_channels: 输入特征维度
            base_channels: 基础通道数
            channels: 每层的通道数列表
            num_blocks: 每层的残差块数量
            sparse_shape: 稀疏形状 [Z, Y, X]
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.sparse_shape = sparse_shape
        
        # 输入层
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=3,
                             stride=1, padding=1, bias=False, indice_key="subm0",
                             algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 编码器（下采样）
        self.encoder_stages = nn.ModuleList()
        in_ch = base_channels
        for i, (out_ch, num_block) in enumerate(zip(channels, num_blocks)):
            stage = nn.ModuleList()
            # 下采样层
            if i == 0:
                stride = 1
            else:
                stride = 2
            
            stage.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(in_ch, out_ch, kernel_size=2, stride=stride,
                                      bias=False, indice_key=f"spconv{i}",
                                      algo=spconv.ConvAlgo.Native),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            # 残差块
            for j in range(num_block):
                stage.append(
                    SparseResidualBlock(
                        out_ch, out_ch, indice_key=f"subm{i}_{j}", stride=1
                    )
                )
            
            self.encoder_stages.append(stage)
            in_ch = out_ch
        
        # 解码器（上采样）
        self.decoder_stages = nn.ModuleList()
        for i in range(len(channels) - 2, -1, -1):
            in_ch = channels[i + 1]
            skip_ch = channels[i] if i > 0 else base_channels
            out_ch = channels[i]
            
            stage = nn.ModuleList()
            # 上采样层
            stage.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(in_ch, out_ch, kernel_size=2,
                                             indice_key=f"spconv{i}", bias=False,
                                             algo=spconv.ConvAlgo.Native),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            # 融合skip connection
            stage.append(
                spconv.SparseSequential(
                    spconv.SubMConv3d(out_ch + skip_ch, out_ch, kernel_size=3,
                                     stride=1, padding=1, bias=False,
                                     indice_key=f"subm_up{i}",
                                     algo=spconv.ConvAlgo.Native),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            # 残差块
            for j in range(num_blocks[i]):
                stage.append(
                    SparseResidualBlock(
                        out_ch, out_ch, indice_key=f"subm_up{i}_{j}", stride=1
                    )
                )
            
            self.decoder_stages.append(stage)
        
        # 输出层
        self.conv_output = spconv.SparseSequential(
            spconv.SubMConv3d(channels[0], channels[0], kernel_size=3,
                            stride=1, padding=1, bias=False, indice_key="subm_out",
                            algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, sparse_tensor):
        """
        Args:
            sparse_tensor: spconv.SparseConvTensor
        
        Returns:
            sparse_tensor: 输出稀疏张量
        """
        # 输入层
        x = self.conv_input(sparse_tensor)
        
        # 编码器
        encoder_features = [x]
        for stage in self.encoder_stages:
            for layer in stage:
                x = layer(x)
            encoder_features.append(x)
        
        # 解码器
        for i, stage in enumerate(self.decoder_stages):
            # 上采样
            x = stage[0](x)
            
            # 融合skip connection
            # skip_idx对应编码器中的特征（从后往前）
            skip_idx = len(self.encoder_stages) - 1 - i
            skip_feat = encoder_features[skip_idx]
            
            # 对于稀疏卷积，上采样后需要对齐skip特征
            # 使用字典快速查找skip特征
            device = x.features.device
            skip_dict = {}
            skip_indices_tensor = skip_feat.indices
            for j in range(skip_indices_tensor.shape[0]):
                idx_key = tuple(skip_indices_tensor[j].cpu().numpy())
                skip_dict[idx_key] = skip_feat.features[j]
            
            # 为x的每个位置查找对应的skip特征
            aligned_skip_feats = []
            x_indices_tensor = x.indices
            skip_feat_dim = skip_feat.features.shape[1]
            
            for j in range(x_indices_tensor.shape[0]):
                idx_key = tuple(x_indices_tensor[j].cpu().numpy())
                if idx_key in skip_dict:
                    aligned_skip_feats.append(skip_dict[idx_key])
                else:
                    # 如果没有找到，使用零特征
                    aligned_skip_feats.append(torch.zeros(skip_feat_dim, device=device, dtype=x.features.dtype))
            
            aligned_skip_feats = torch.stack(aligned_skip_feats, dim=0)
            x = x.replace_feature(
                torch.cat([x.features, aligned_skip_feats], dim=1)
            )
            
            x = stage[1](x)
            
            # 残差块
            for j in range(2, len(stage)):
                x = stage[j](x)
        
        # 输出层
        x = self.conv_output(x)
        
        return x


class VoxelToPointMapper(nn.Module):
    """
    将voxel特征映射回点云。
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, voxel_features, point_to_voxel_map, num_points):
        """
        将voxel特征映射回点云。
        
        Args:
            voxel_features: [M, C] voxel特征
            point_to_voxel_map: [N] 每个点对应的voxel索引，-1表示无效点
            num_points: 原始点云数量
        
        Returns:
            point_features: [N, C] 点云特征
        """
        device = voxel_features.device
        M, C = voxel_features.shape
        
        # 创建点云特征张量
        point_features = torch.zeros(num_points, C, device=device, dtype=voxel_features.dtype)
        
        # 有效点的mask
        valid_mask = point_to_voxel_map >= 0
        valid_indices = point_to_voxel_map[valid_mask]
        
        # 将voxel特征映射到点
        point_features[valid_mask] = voxel_features[valid_indices]
        
        return point_features


class SparseUNetSegModel(nn.Module):
    """
    完整的稀疏3D卷积UNet分割模型。
    流程：点云 -> voxelization -> UNet -> MaskFormer -> 恢复点云
    """
    
    def __init__(
        self,
        voxel_size=0.05,
        point_cloud_range=None,
        in_channels=3,
        base_channels=16,
        unet_channels=[16, 32, 64, 128, 256],
        unet_num_blocks=[2, 2, 2, 2, 2],
        num_queries=32,
        maskformer_n_layers=2,
        maskformer_embed_dim=256,
        maskformer_n_head=8,
        maskformer_hidden_dim=256,
        use_pos_enc=False
    ):
        """
        Args:
            voxel_size: voxel大小
            point_cloud_range: 点云范围
            in_channels: 输入特征维度
            base_channels: UNet基础通道数
            unet_channels: UNet每层通道数
            unet_num_blocks: UNet每层残差块数量
            num_queries: MaskFormer查询数量
            maskformer_n_layers: MaskFormer transformer层数
            maskformer_embed_dim: MaskFormer embedding维度
            maskformer_n_head: MaskFormer attention头数
            maskformer_hidden_dim: MaskFormer隐藏层维度
            use_pos_enc: 是否使用位置编码
        """
        super().__init__()
        
        # Voxelization模块
        self.voxelizer = PointVoxelizer(voxel_size, point_cloud_range)
        
        # 计算sparse_shape（如果提供了point_cloud_range）
        if point_cloud_range is not None:
            sparse_shape = [
                int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size),
                int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size),
                int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size)
            ]
        else:
            sparse_shape = None
        
        # 稀疏3D卷积UNet
        self.unet = SparseUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            channels=unet_channels,
            num_blocks=unet_num_blocks,
            sparse_shape=sparse_shape
        )
        
        # Voxel到点云的映射器
        self.voxel_to_point = VoxelToPointMapper()
        
        # MaskFormer分割头
        self.maskformer_head = MaskFormerHead(
            in_channels=unet_channels[0],
            num_queries=num_queries,
            n_transformer_layer=maskformer_n_layers,
            transformer_embed_dim=maskformer_embed_dim,
            transformer_n_head=maskformer_n_head,
            transformer_hidden_dim=maskformer_hidden_dim,
            use_pos_enc=use_pos_enc
        )
    
    def forward(self, point_dict, return_teacher=False, upsample_masks=False):
        """
        前向传播。
        
        Args:
            point_dict: 包含点云数据的字典
                - coord: [N, 3] 点云坐标
                - feat: [N, C] 点云特征（可选，默认使用坐标）
                - batch: [N] batch索引
            return_teacher: 不使用（保持接口兼容）
            upsample_masks: 不使用（保持接口兼容）
        
        Returns:
            Dictionary with:
                - upsampled_student_masks: [K, N] 分割mask
                - upsampled_batch: [N] batch索引
        """
        coords = point_dict['coord']  # [N, 3]
        feats = point_dict.get('feat', coords)  # [N, C] 或 [N, 3]
        batch = point_dict['batch']  # [N]
        
        # 1. Voxelization，保存映射关系
        voxel_coords, voxel_features, point_to_voxel_map, voxel_to_point_map = \
            self.voxelizer(coords, batch)
        
        if voxel_coords.shape[0] == 0:
            # 如果没有有效voxel，返回空mask
            num_queries = self.maskformer_head.num_queries
            empty_masks = torch.zeros(num_queries, coords.shape[0], 
                                     device=coords.device, dtype=coords.dtype)
            return {
                'upsampled_student_masks': empty_masks,
                'upsampled_batch': batch
            }
        
        # 2. 创建稀疏卷积张量
        batch_size = batch.max().item() + 1
        
        # 计算sparse_shape（如果未提供）
        if self.unet.sparse_shape is None:
            voxel_max = voxel_coords[:, 1:].max(dim=0)[0]
            sparse_shape = [voxel_max[0].item() + 1, 
                          voxel_max[1].item() + 1, 
                          voxel_max[2].item() + 1]
        else:
            sparse_shape = self.unet.sparse_shape
        
        sparse_tensor = spconv.SparseConvTensor(
            features=voxel_features.contiguous(),
            indices=voxel_coords.contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch_size
        )
        
        # 3. 通过UNet
        unet_output = self.unet(sparse_tensor)
        
        # 4. 将voxel特征映射回点云
        point_features = self.voxel_to_point(
            unet_output.features,
            point_to_voxel_map,
            coords.shape[0]
        )
        
        # 5. 通过MaskFormer分割头
        # 创建point对象（用于MaskFormer）
        class Point:
            def __init__(self, batch):
                self.batch = batch
        
        point_obj = Point(batch)
        masks = self.maskformer_head(point_features, coords, point_obj)
        
        return {
            'upsampled_student_masks': masks,
            'upsampled_batch': batch
        }
    
    def update_teacher(self):
        """占位符（保持接口兼容）"""
        pass

