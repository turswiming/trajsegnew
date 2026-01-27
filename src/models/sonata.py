"""
Sonata model wrapper for Point Transformer V3.
Supports both Sonata and Concerto pretrained weights.
"""

import torch
import torch.nn as nn
import sys
import os

# Try to import Sonata model
SONATA_AVAILABLE = True
# Try multiple possible paths
possible_paths = [
    '/workspace/sonata_pt_reproduction/sonata',
    '/workspace/sonata_pt_reproduction',
]
for path in possible_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
from sonata.model import PointTransformerV3, load as sonata_load
SONATA_AVAILABLE = True


class SonataModel(nn.Module):
    """
    Sonata model wrapper for scene flow estimation.
    Uses Point Transformer V3 as backbone.
    Supports both Sonata and Concerto pretrained weights.
    """

    def __init__(
        self,
        use_pretrained=True,
        pretrained_name="sonata",
        pretrained_path=None,
        enc_mode=True,
        in_channels=6,
        **kwargs
    ):
        """
        Initialize Sonata model.

        Args:
            use_pretrained: Whether to use pre-trained weights
            pretrained_name: Name of pre-trained model ("sonata", "sonata_small", "concerto_outdoor", "concerto_indoor")
            pretrained_path: Path to local pretrained checkpoint (for Concerto)
            enc_mode: Whether to use encoder-only mode
            in_channels: Input channels (coord + features)
        """
        super().__init__()
        
        if not SONATA_AVAILABLE:
            raise ImportError(
                "Sonata model is not available. Please install sonata package and its dependencies."
            )

        if use_pretrained:
            if pretrained_name.startswith("concerto"):
                # Load Concerto pretrained weights
                self._load_concerto(pretrained_path, pretrained_name, enc_mode)
            else:
                # Load Sonata pretrained weights from HuggingFace
                self._load_sonata(pretrained_name, enc_mode)
        else:
            self.backbone = PointTransformerV3(
                in_channels=in_channels,
                enc_mode=enc_mode,
                enc_depths=(3, 3, 3, 12, 3),
                enc_channels=(48, 96, 192, 384, 512),
                enc_num_head=(3, 6, 12, 24, 32),
                enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                **kwargs
            )
    
    def _load_concerto(self, pretrained_path, pretrained_name, enc_mode):
        """Load Concerto pretrained weights from local path."""
        if pretrained_path is None:
            # Default path based on pretrained_name
            if pretrained_name == "concerto_outdoor":
                pretrained_path = "checkpoints/pretrained/concerto_large_outdoor.pth"
            elif pretrained_name == "concerto_indoor":
                pretrained_path = "checkpoints/pretrained/concerto_large_indoor.pth"
            else:
                raise ValueError(f"Unknown Concerto model: {pretrained_name}")
        
        print(f"Loading Concerto model from: {pretrained_path}")
        
        # Load checkpoint
        ckpt = torch.load(pretrained_path, map_location='cpu')
        config = ckpt['config']
        state_dict = ckpt['state_dict']
        
        # Print config for debugging
        print(f"Concerto config: in_channels={config['in_channels']}, "
              f"enc_channels={config['enc_channels']}, enc_mode={config.get('enc_mode', enc_mode)}")
        
        # Create model with the same config
        # Force enc_mode to be True if specified
        if enc_mode:
            config['enc_mode'] = True
        
        self.backbone = PointTransformerV3(**config)
        
        # Load state dict
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"Concerto model loaded successfully! Output dim: {config['enc_channels'][-1]}")
        
        # Store the output dimension for later use
        self.output_dim = config['enc_channels'][-1]
        self.in_channels = config['in_channels']
    
    def _load_sonata(self, pretrained_name, enc_mode):
        """Load Sonata pretrained weights from HuggingFace."""
        try:
            print(f"Loading pre-trained Sonata model: {pretrained_name}")
            self.backbone = sonata_load(pretrained_name, repo_id="facebook/sonata")
            if enc_mode and not hasattr(self.backbone, 'enc_mode'):
                self.backbone.enc_mode = True
            
            # Get output dimension from config
            if hasattr(self.backbone, 'enc'):
                # Get the last encoder stage's output channels
                enc_channels = [48, 96, 192, 384, 512]  # Default Sonata channels
                self.output_dim = enc_channels[-1]
            else:
                self.output_dim = 512
            
            self.in_channels = getattr(self.backbone.embedding, 'in_channels', 6)
            
        except Exception as e:
            print(f"Warning: Could not load pre-trained model: {e}")
            print("Using randomly initialized Point Transformer V3 instead.")
            self.backbone = PointTransformerV3(
                in_channels=6,
                enc_mode=enc_mode,
                enc_depths=(3, 3, 3, 12, 3),
                enc_channels=(48, 96, 192, 384, 512),
                enc_num_head=(3, 6, 12, 24, 32),
                enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            )
            self.output_dim = 512
            self.in_channels = 6

    def forward(self, point_dict):
        """
        Forward pass.

        Args:
            point_dict: Dictionary with keys:
                - coord: [N, 3] point coordinates
                - feat: [N, C] point features (optional)
                - batch: [N] batch indices (optional)
                - grid_size: Grid size for serialization (optional, default: 0.1)

        Returns:
            Point object with features
        """
        # Create a copy to avoid modifying the original dict
        point_dict = point_dict.copy()
        
        # Add grid_size if not provided (required for serialization)
        if 'grid_size' not in point_dict:
            point_dict['grid_size'] = 0.1
        
        # Get model's expected input channels
        model_in_channels = getattr(self, 'in_channels', 6)
        
        # Prepare features based on model's expected input channels
        if 'feat' in point_dict:
            feat = point_dict['feat']
        else:
            feat = point_dict['coord']
        
        feat_channels = feat.shape[1]
        
        # Adjust feature dimensions to match model's expected input
        if feat_channels < model_in_channels:
            # Pad features with zeros
            padding_size = model_in_channels - feat_channels
            padding = torch.zeros(
                feat.shape[0], 
                padding_size, 
                device=feat.device, 
                dtype=feat.dtype
            )
            feat = torch.cat([feat, padding], dim=-1)
        elif feat_channels > model_in_channels:
            # Truncate features if too many channels
            feat = feat[:, :model_in_channels]
        
        # Update point_dict with adjusted features
        point_dict['feat'] = feat
        
        return self.backbone(point_dict)
    
    def get_output_dim(self):
        """Get the output feature dimension of the encoder."""
        return getattr(self, 'output_dim', 512)
