"""
Scene flow prediction head.
"""

import torch
import torch.nn as nn


class FlowHead(nn.Module):
    """
    MLP head for scene flow prediction from point features.
    """

    def __init__(self, in_channels=512, hidden_channels=256, out_channels=3):
        """
        Initialize flow head.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (3 for scene flow)
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, features):
        """
        Predict scene flow from features.

        Args:
            features: [N, in_channels] point features

        Returns:
            flow: [N, 3] predicted scene flow
        """
        return self.mlp(features)

