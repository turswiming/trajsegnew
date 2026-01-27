"""
Instance segmentation head for point cloud segmentation.
"""

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """
    MLP head for instance segmentation from point features.
    Outputs K segmentation masks (one for each instance).
    """

    def __init__(self, in_channels=512, hidden_channels=256, num_instances=4):
        """
        Initialize segmentation head.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            num_instances: Number of instance masks K
        """
        super().__init__()
        
        self.num_instances = num_instances
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, num_instances)
        )

    def forward(self, features):
        """
        Predict instance segmentation masks from features.

        Args:
            features: [N, in_channels] point features

        Returns:
            logits: [N, num_instances] segmentation logits (before softmax)
        """
        logits = self.mlp(features)  # [N, num_instances]
        return logits




