"""
Instance segmentation model for point clouds with sceneflow supervision.
"""

import torch
import torch.nn as nn
from .sonata import SonataModel
from .segmentation_head import SegmentationHead


class SegmentationModel(nn.Module):
    """
    Model for instance segmentation with sceneflow supervision.
    Input: single frame point cloud
    Output: instance segmentation masks
    Supervision: sceneflow (from dataset)
    """

    def __init__(
        self,
        use_pretrained=True,
        pretrained_name="sonata",
        feature_dim=512,
        num_instances=4,
        in_channels=3,
    ):
        """
        Initialize segmentation model.

        Args:
            use_pretrained: Whether to use pre-trained Sonata weights
            pretrained_name: Name of pre-trained model
            feature_dim: Feature dimension from encoder
            num_instances: Number of instance masks K
            in_channels: Input channels (3 for xyz coordinates)
        """
        super().__init__()
        
        # Encoder
        self.encoder = SonataModel(
            use_pretrained=use_pretrained,
            pretrained_name=pretrained_name,
            enc_mode=True,
            in_channels=in_channels
        )
        
        # Segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels=feature_dim,
            hidden_channels=256,
            num_instances=num_instances
        )

    def forward(self, point_dict):
        """
        Forward pass.

        Args:
            point_dict: Dictionary with keys:
                - coord: [N, 3] point coordinates
                - feat: [N, C] point features (optional, defaults to coords)
                - batch: [N] batch indices (optional)

        Returns:
            Dictionary with:
                - features: Encoded point features
                - segmentation_logits: [N, K] segmentation logits
                - segmentation_masks: [K, N] segmentation masks (softmax applied)
        """
        # Encode point cloud
        encoded_features = self.encoder(point_dict)
        
        # Get point features
        point_features = encoded_features.feat  # [N, feature_dim]
        
        # Predict segmentation
        segmentation_logits = self.segmentation_head(point_features)  # [N, K]
        
        # Apply softmax to get masks
        segmentation_masks = torch.softmax(segmentation_logits, dim=1)  # [N, K]
        segmentation_masks = segmentation_masks.permute(1, 0)  # [K, N]
        
        return {
            'features': encoded_features,
            'segmentation_logits': segmentation_logits,
            'segmentation_masks': segmentation_masks
        }




