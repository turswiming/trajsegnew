"""
Flow Smoothness Loss implementation for scene flow prediction.

This module implements a parametric flow smoothness loss that encourages spatially
coherent flow predictions within segmented regions. It uses a quadratic flow
approximation approach to ensure smooth transitions in the predicted flow field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ScaleGradient(torch.autograd.Function):
    """
    Custom autograd function for gradient scaling during backpropagation.
    """

    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

class BinaryMask(torch.autograd.Function):
    """
    Custom autograd function for binary mask using Straight-Through Estimator.
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

class FlowSmoothLoss(nn.Module):
    """
    Flow smoothness loss using parametric quadratic flow approximation.

    This loss encourages smooth flow fields within segmented regions by fitting
    a quadratic model to the flow in each segment and penalizing deviations
    from this model.
    """

    def __init__(
        self,
        device,
        each_mask_item_gradient=0.5,
        sum_mask_item_gradient=0.5,
        each_mask_item_loss="L2",
        sum_mask_item_loss="L2",
        scale_flow_grad=1.0,
        square_mask=False,
        normalize_flow=True,
        singular_value_loss_gradient=0.0,
        sparse_filter_ratio=0.0,
        use_checkpoint=False,
        downsample_ratio=1,
    ):
        """
        Initialize the Flow Smoothness Loss.

        Args:
            device: Device to perform computations on
            each_mask_item_gradient: Weight for per-mask reconstruction loss
            sum_mask_item_gradient: Weight for total reconstruction loss
            each_mask_item_loss: Loss type for per-mask ("L1" or "L2")
            sum_mask_item_loss: Loss type for total ("L1" or "L2")
            scale_flow_grad: Gradient scaling factor for flow
            square_mask: Whether to square root the mask
            normalize_flow: Whether to normalize flow before computing loss
            singular_value_loss_gradient: Weight for singular value loss
            sparse_filter_ratio: Ratio of points to filter out (robust loss)
            use_checkpoint: Whether to use gradient checkpointing to save memory
            downsample_ratio: Downsample ratio for the point cloud
        """
        super().__init__()
        self.device = device
        self.use_checkpoint = use_checkpoint
        
        # Normalize gradients
        total_grad = each_mask_item_gradient + sum_mask_item_gradient
        if total_grad > 0:
            self.each_mask_item_gradient = each_mask_item_gradient / total_grad
            self.sum_mask_item_gradient = sum_mask_item_gradient / total_grad
        else:
            self.each_mask_item_gradient = 0.0
            self.sum_mask_item_gradient = 0.0
        
        self.scale_flow_grad = scale_flow_grad
        self.square_mask = square_mask
        self.normalize_flow = normalize_flow
        self.singular_value_loss_gradient = singular_value_loss_gradient
        
        # Setup loss criteria
        if each_mask_item_loss in ["L1", "l1"]:
            self.each_mask_criterion = nn.L1Loss(reduction="none")
        elif each_mask_item_loss in ["L2", "l2"]:
            self.each_mask_criterion = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Invalid loss criterion: {each_mask_item_loss}")
        
        if sum_mask_item_loss in ["L1", "l1"]:
            self.sum_mask_criterion = nn.L1Loss(reduction="none")
        elif sum_mask_item_loss in ["L2", "l2"]:
            self.sum_mask_criterion = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Invalid loss criterion: {sum_mask_item_loss}")
        self.downsample_ratio = downsample_ratio

    def forward(self, point_position, mask, flow, singular_value_loss=False):
        """
        Compute the flow smoothness loss.

        Args:
            point_position: List of point positions, each of shape [N, 3]
            mask: List of segmentation masks, each of shape [K, N]
            flow: List of predicted flow vectors, each of shape [N, 3]
            singular_value_loss: Whether to compute singular value loss

        Returns:
            Computed smoothness loss averaged across the batch
        """
        batch_size = len(point_position)
        point_position = [item.to(self.device) for item in point_position]
        scene_flows = flow

        #根据sceneflow的幅度筛选top-10%的点去计算
        magnitudes = [torch.norm(sf, dim=-1) for sf in scene_flows]
        magnitude_masks = []
        for mag in magnitudes:
            threshold = torch.quantile(mag, 0.90)
            magnitude_mask = mag >= threshold
            magnitude_masks.append(magnitude_mask)
            pass

        total_loss = torch.tensor(0.0, device=self.device)
        for b in range(batch_size):
            # Use gradient checkpoint for each sample to save memory
            if self.use_checkpoint:
                batch_loss = checkpoint(
                    self._process_single_sample,
                    point_position[b][magnitude_masks[b]][::self.downsample_ratio],
                    mask[b][:,magnitude_masks[b]][:,::self.downsample_ratio],
                    scene_flows[b][magnitude_masks[b]][::self.downsample_ratio],
                    torch.tensor(singular_value_loss, device=self.device),
                    use_reentrant=False
                )
            else:
                batch_loss = self._process_single_sample(
                    point_position[b][magnitude_masks[b]][::self.downsample_ratio],
                    mask[b][:,magnitude_masks[b]][:,::self.downsample_ratio],
                    scene_flows[b][magnitude_masks[b]][::self.downsample_ratio],
                    torch.tensor(singular_value_loss, device=self.device)
                )
            total_loss = total_loss + batch_loss

        return total_loss / batch_size

    def _process_single_sample(self, point_position_b, mask_b, scene_flow_b, singular_value_loss_flag):
        """
        Process a single sample for flow smoothness computation.
        This function is designed to be used with gradient checkpointing.
        
        Args:
            point_position_b: Point positions [N, 3]
            mask_b: Segmentation masks [K, N]
            scene_flow_b: Scene flow [N, 3]
            singular_value_loss_flag: Whether to compute singular value loss (tensor)
        
        Returns:
            Total loss for this sample
        """
        N = point_position_b.shape[0]
        K = mask_b.shape[0]
        
        # mask_binary_b = BinaryMask.apply(mask_b)
        mask_binary_b = mask_b
        if self.normalize_flow:
            flow_std = scene_flow_b.std(dim=0).detach().max()
            scene_flow_b = scene_flow_b / (flow_std + 1e-1) * 10

        
        # Construct embedding
        coords = self._construct_embedding(point_position_b)  # (N, 4)

        # Initialize
        flow_reconstruction = torch.zeros_like(scene_flow_b)  # (N, 3)
        one_batch_loss = torch.tensor(0.0, device=self.device)
        no_scale_loss = torch.tensor(0.0, device=self.device)

        for k in range(K):
            mask_bk = mask_binary_b[k]  # (N,)
            if mask_bk.dim() > 1:
                mask_bk = mask_bk.squeeze()
            if self.square_mask:
                mask_bk = torch.sqrt(mask_bk)

            Ek = coords * mask_bk.unsqueeze(-1)  # (N, 4)
            Fk = scene_flow_b * mask_bk.unsqueeze(-1)  # (N, 3)
            Magnitude_Fk = torch.norm(Fk, dim=-1)
            # Mask_indices = mask_bk>0
            # Ek = Ek[Mask_indices]
            # Fk = Fk[Mask_indices]
            theta = torch.linalg.lstsq(Ek, Fk, driver="gels").solution  # (4, 3)
            
            if singular_value_loss_flag.item() and self.singular_value_loss_gradient > 0:
                M = theta[:3, :].T + torch.eye(3, device=self.device)
                U, S, V_h = torch.linalg.svd(M, full_matrices=False)
                no_scale_loss = no_scale_loss + (
                    F.mse_loss(S, torch.ones_like(S, device=self.device))
                    * self.singular_value_loss_gradient
                )
            
            if torch.isnan(theta).any():
                flow_reconstruction = flow_reconstruction + Fk
                continue
                
            Fk_hat = Ek @ theta  # (N_filtered, 3)
            
            # Update flow reconstruction
            flow_reconstruction = flow_reconstruction + Fk_hat

            # Per-mask loss
            if self.each_mask_item_gradient > 0:
                batch_reconstruction_loss = self.each_mask_criterion(
                    Fk_hat, Fk
                ).mean(dim=-1)
                
                one_batch_loss = one_batch_loss + batch_reconstruction_loss.sum() * self.each_mask_item_gradient / N
        
        # Total reconstruction loss
        if self.sum_mask_item_gradient > 0:
            reconstruction_loss = self.sum_mask_criterion(
                scene_flow_b, flow_reconstruction
            )  # (N, 3)            
            one_batch_loss = one_batch_loss + reconstruction_loss.sum() * self.sum_mask_item_gradient / N
        
        return one_batch_loss + no_scale_loss

    def _construct_embedding(self, point_position):
        """
        Construct point coordinate embedding [x, y, z, 1].

        Args:
            point_position: Point cloud positions [N, 3]

        Returns:
            Embedding vectors [N, 4]
        """
        x = point_position[..., 0].view(-1)
        y = point_position[..., 1].view(-1)
        z = point_position[..., 2].view(-1)
        
        emb = torch.stack([x, y, z, torch.ones_like(x)], dim=1)
        return emb
