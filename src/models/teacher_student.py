"""
Teacher-Student model for self-supervised learning.
"""

import torch
import torch.nn as nn
import copy
from .sonata import SonataModel
from .maskformer_head import MaskFormerHead


def update_teacher_ema(student, teacher, alpha=0.999):
    """
    Update teacher model using exponential moving average.

    Args:
        student: Student model
        teacher: Teacher model
        alpha: EMA decay factor
    """
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data


class TeacherStudentModel(nn.Module):
    """
    Teacher-Student model for self-supervised scene flow learning.
    """

    def __init__(
        self,
        use_pretrained=True,
        pretrained_name="sonata",
        pretrained_path=None,
        feature_dim=512,
        num_queries=4,
        ema_decay=0.999,
        freeze_encoder=False,
    ):
        """
        Initialize Teacher-Student model.

        Args:
            use_pretrained: Whether to use pre-trained Sonata weights
            pretrained_name: Name of pre-trained model ("sonata", "sonata_small", "concerto_outdoor", "concerto_indoor")
            pretrained_path: Path to local pretrained checkpoint (for Concerto)
            feature_dim: Feature dimension from encoder (will be auto-detected if using pretrained)
            num_queries: Number of mask queries for MaskFormer
            ema_decay: EMA decay factor for teacher update
            freeze_encoder: Whether to freeze encoder weights (only train MaskFormer head)
        """
        super().__init__()
        
        self.freeze_encoder = freeze_encoder
        
        # Student model
        self.student_encoder = SonataModel(
            use_pretrained=use_pretrained,
            pretrained_name=pretrained_name,
            pretrained_path=pretrained_path,
            enc_mode=True
        )
        
        # Get the actual output dimension from the encoder
        actual_feature_dim = self.student_encoder.get_output_dim()
        print(f"Encoder output dimension: {actual_feature_dim}")
        
        # Use the actual feature dimension for MaskFormer head
        self.student_maskformer_head = MaskFormerHead(
            in_channels=actual_feature_dim,
            num_queries=num_queries,
            n_transformer_layer=2,
            transformer_embed_dim=256,
            transformer_n_head=8,
            transformer_hidden_dim=256,
            use_pos_enc=True
        )
        
        # Teacher model (only encoder, no segmentation head)
        self.teacher_encoder = SonataModel(
            use_pretrained=use_pretrained,
            pretrained_name=pretrained_name,
            pretrained_path=pretrained_path,
            enc_mode=True
        )
        
        # Initialize teacher encoder with student encoder weights
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        
        # Freeze teacher (no gradients)
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        
        # Freeze student encoder if specified (only train MaskFormer head)
        if freeze_encoder:
            print("Freezing encoder weights - only training MaskFormer head")
            for param in self.student_encoder.parameters():
                param.requires_grad = False
        
        self.ema_decay = ema_decay
        
        # Print trainable parameters info
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Print the number of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"Frozen parameters: {frozen_params / 1e6:.2f}M")

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher model using EMA (only encoder, no segmentation head)."""
        if not self.freeze_encoder:
            # Only update if encoder is being trained
            update_teacher_ema(self.student_encoder, self.teacher_encoder, self.ema_decay)

    def forward(self, point_dict, return_teacher=True, upsample_masks=False):
        """
        Forward pass through student and teacher models.

        Args:
            point_dict: Dictionary with point cloud data (batch of 4 point clouds)
            return_teacher: Whether to return teacher predictions
            upsample_masks: If True, upsample masks to original scale (for validation)
                           If False, return masks at downsampled scale (for training, saves memory)

        Returns:
            Dictionary with:
            - student_masks: [K, N_down] masks at downsampled scale (or [K, N_total] if upsample_masks=True)
            - student_features: Point object with encoded features (for consistency loss)
            - teacher_features: Point object with encoded features (for consistency loss)
            - down_coords: [N_down, 3] downsampled coordinates
            - down_batch: [N_down] batch indices for downsampled points
            - student_batch: [N_total] original batch indices (for upsampled masks)
        """
        # Save original batch indices
        batch = point_dict['batch']  # [N_total] batch indices for original points
        
        # Deep copy point_dict for teacher BEFORE student encoder (student may modify it)
        teacher_point_dict = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in point_dict.items()
        }
        
        # Student forward: encoder + segmentation head
        student_points = self.student_encoder(point_dict)
        
        # Get downsampled coordinates and batch indices
        down_coords = student_points.coord  # [N_down, 3]
        down_batch = student_points.batch   # [N_down]
        
        student_masks = self.student_maskformer_head(
            student_points.feat, point=student_points,point_coords=down_coords
        )  # [K, N_down]
        
        # Compute downsample indices: down_indices[i] tells which downsampled point
        # the i-th original point maps to. So: upsampled_masks[:, i] = student_masks[:, down_indices[i]]
        down_indices = self._compute_downsample_indices_v3(student_points)  # [N_total]
        upsampled_masks = self._upsample_masks(student_points, student_masks)

        result = {
            'student_features': student_points,  # Point object, for consistency loss
            'downsampled_student_masks': student_masks,      # [K, N_down] masks at downsampled scale
            'upsampled_student_masks': upsampled_masks,      # [K, N_total] masks at original scale
            'upsampled_batch': batch,                       # [N_total] batch indices for upsampled points
            'downsampled_batch': down_batch,                # [N_down] batch indices for downsampled points
            'down_indices': down_indices,                  # [N_down] indices into original point cloud
        }
        
        # Teacher forward: only encoder, no segmentation head
        if return_teacher:
            result['teacher_features'] = student_points
            # with torch.no_grad():
            #     teacher_features = self.teacher_encoder(teacher_point_dict)
            #     result['teacher_features'] = teacher_features
        
        return result
    
    def _upsample_masks(self, point, masks):
        """
        Upsample masks from encoded scale to original scale following Sonata README (162-181).
        Strictly follows the README method without padding or error handling.
        
        Args:
            point: Point object from encoder (contains batch, pooling_parent, pooling_inverse)
            masks: [K, N_encoded] mask logits at encoded scale
            original_batch: [N_original] batch tensor from original point_dict
        
        Returns:
            upsampled_masks: [K, N_original] upsampled mask logits
            upsampled_batch: [N_original] batch tensor for upsampled points
        """
        # Strictly follow README method:
        # Step 1: First 2 levels with concatenation (for masks, we use indexing)
        # Step 2: Remaining levels with simple indexing
        # Step 3: Final mapping using point.inverse
        
        current_point = point
        current_masks = masks  # [K, N_encoded]
        # First 2 levels: following README exactly
        # README: for _ in range(2): parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        # For masks: we use direct indexing (no concatenation needed for masks)
        pooling_indices = []
        for _ in range(2):
            assert "pooling_parent" in current_point.keys()
            assert "pooling_inverse" in current_point.keys()
            parent = current_point["pooling_parent"]
            inverse = current_point["pooling_inverse"]
            pooling_indice =  current_point["pooling_indices"]
            # For masks: index on the point dimension (dim=1), not the mask dimension (dim=0)
            current_masks = current_masks[:, inverse]
            current_point = parent
        
        # Remaining levels: following README exactly
        # README: while "pooling_parent" in point.keys(): parent.feat = point.feat[inverse]
        while "pooling_parent" in current_point.keys():
            assert "pooling_inverse" in current_point.keys()
            
            parent = current_point["pooling_parent"]
            inverse = current_point["pooling_inverse"]
            
            # For masks: index on the point dimension (dim=1), not the mask dimension (dim=0)
            current_masks = current_masks[:, inverse]
            current_point = parent
        
        return current_masks
    
    
    @torch.no_grad()
    def _rebuild_pooling_indices_from_inverse(self, pooling_inverse, n_down):
        """
        Rebuild pooling_indices from pooling_inverse.
        
        pooling_inverse: [N_up] tells us upsampled[i] = downsampled[pooling_inverse[i]]
        pooling_indices: [N_down] tells us downsampled[j] came from upsampled[pooling_indices[j]]
        
        For each downsampled point j, find the first i where pooling_inverse[i] == j.
        
        Args:
            pooling_inverse: [N_up] indices mapping upsampled points to downsampled points
            n_down: number of downsampled points
        
        Returns:
            pooling_indices: [N_down] indices into the upsampled point cloud
        """
        device = pooling_inverse.device
        n_up = pooling_inverse.shape[0]
        
        # For each downsampled point j, find the first i where pooling_inverse[i] == j
        # We'll use scatter to find the minimum index for each downsampled point
        pooling_indices = torch.full((n_down,), n_up, device=device, dtype=torch.long)
        
        # Create indices [0, 1, 2, ..., n_up-1]
        up_indices = torch.arange(n_up, device=device)
        
        # For each position in pooling_inverse, record the minimum index
        # scatter_reduce with 'amin' finds the minimum index that maps to each downsampled point
        pooling_indices.scatter_reduce_(0, pooling_inverse, up_indices, reduce='amin', include_self=False)
        
        return pooling_indices
    
    @torch.no_grad()
    def _compute_downsample_indices_v3(self, point):
        """
        Compute indices for downsampling by rebuilding pooling_indices from pooling_inverse.
        
        Returns:
            downsample_indices: [N_down] indices into original point cloud,
                               meaning downsampled_point[i] came from original_point[downsample_indices[i]]
        """
        # Collect all levels' info and rebuild pooling_indices
        levels_info = []  # List of (pooling_inverse, n_down) for each level
        current_point = point
        
        # First 2 levels
        for _ in range(2):
            if "pooling_parent" not in current_point.keys():
                break
            inverse = current_point["pooling_inverse"]
            parent = current_point["pooling_parent"]
            n_down = current_point.coord.shape[0]
            levels_info.append((inverse, n_down))
            current_point = parent
        
        # Remaining levels
        while "pooling_parent" in current_point.keys():
            inverse = current_point["pooling_inverse"]
            parent = current_point["pooling_parent"]
            n_down = current_point.coord.shape[0]
            levels_info.append((inverse, n_down))
            current_point = parent
        
        if len(levels_info) == 0:
            # No downsampling happened
            N_down = point.coord.shape[0]
            return torch.arange(N_down, device=point.coord.device)
        
        # Rebuild pooling_indices for each level
        all_pooling_indices = []
        for inverse, n_down in levels_info:
            pooling_indices = self._rebuild_pooling_indices_from_inverse(inverse, n_down)
            all_pooling_indices.append(pooling_indices)
        
        # Compose: start from the last level (closest to original)
        # all_pooling_indices[-1] maps from level[-1] to original
        composed_indices = all_pooling_indices[-1]
        
        # Compose backwards through all levels
        for i in range(len(all_pooling_indices) - 2, -1, -1):
            # all_pooling_indices[i] maps from level[i] to level[i+1]
            # composed_indices currently maps from level[i+1] to original
            # We need: composed_indices[all_pooling_indices[i]] to map from level[i] to original
            composed_indices = composed_indices[all_pooling_indices[i]]
        
        return composed_indices  # [N_down] indices into original point cloud
    
    def downsample_from_upsampled(self, upsampled_masks, down_indices, n_down):
        """
        Downsample masks from upsampled scale back to downsampled scale.
        This is the inverse operation of upsampling via down_indices.
        
        Since multiple original points map to the same downsampled point,
        we use scatter_mean to aggregate values.
        
        Args:
            upsampled_masks: [K, N_total] masks at original scale
            down_indices: [N_total] indices mapping original points to downsampled points
            n_down: Number of downsampled points
        
        Returns:
            downsampled_masks: [K, N_down] masks at downsampled scale
        
        Example:
            # upsampled_masks[:, i] came from student_masks[:, down_indices[i]]
            # This function reverses that by averaging all points that map to the same down point
            downsampled = model.downsample_from_upsampled(upsampled_masks, down_indices, n_down)
        """
        K = upsampled_masks.shape[0]
        device = upsampled_masks.device
        
        # Initialize output with zeros
        downsampled_masks = torch.zeros(K, n_down, device=device, dtype=upsampled_masks.dtype)
        
        # Count how many original points map to each downsampled point
        counts = torch.zeros(n_down, device=device)
        counts.scatter_add_(0, down_indices, torch.ones_like(down_indices, dtype=counts.dtype))
        
        # Accumulate values for each downsampled point
        # down_indices: [N_total], need to expand to [K, N_total] for scatter_add
        expanded_indices = down_indices.unsqueeze(0).expand(K, -1)  # [K, N_total]
        downsampled_masks.scatter_add_(1, expanded_indices, upsampled_masks)
        
        # Average by dividing by counts (avoid division by zero)
        counts = counts.clamp(min=1)
        downsampled_masks = downsampled_masks / counts.unsqueeze(0)
        
        return downsampled_masks
