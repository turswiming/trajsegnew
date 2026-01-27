"""
MaskFormer segmentation head for point cloud instance segmentation.
Adapted from MaskFormer (NeurIPS'21) for point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer for MaskFormer."""
    
    def __init__(self, embed_dim=256, n_head=8, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=False)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, query, key_value, pos_enc=None):
        """
        Args:
            query: [K, N, D] query embeddings (K queries, N points)
            key_value: [N, D] point features
            pos_enc: [N, D] positional encoding (optional)
        Returns:
            [K, N, D] updated query embeddings
        """
        # Self-attention
        q = query
        attn_out, _ = self.self_attn(q, q, q)
        q = self.norm1(q + attn_out)
        
        # Cross-attention
        kv = key_value.unsqueeze(0).expand(q.shape[0], -1, -1)  # [K, N, D]
        if pos_enc is not None:
            kv = kv + pos_enc.unsqueeze(0)
        
        attn_out, _ = self.cross_attn(q, kv, kv)
        q = self.norm2(q + attn_out)
        
        # FFN
        ffn_out = self.ffn(q)
        q = self.norm3(q + ffn_out)
        
        return q


class MaskFormerHead(nn.Module):
    """
    MaskFormer segmentation head for point cloud instance segmentation.
    Uses transformer decoder to generate mask queries and outputs segmentation masks.
    """
    
    def __init__(
        self,
        in_channels=512,
        num_queries=4,
        n_transformer_layer=2,
        transformer_embed_dim=256,
        transformer_n_head=8,
        transformer_hidden_dim=256,
        use_pos_enc=False
    ):
        """
        Initialize MaskFormer head.
        
        Args:
            in_channels: Input feature dimension
            num_queries: Number of mask queries (K)
            n_transformer_layer: Number of transformer decoder layers
            transformer_embed_dim: Transformer embedding dimension
            transformer_n_head: Number of attention heads
            transformer_hidden_dim: Hidden dimension in FFN
            use_pos_enc: Whether to use positional encoding
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.transformer_embed_dim = transformer_embed_dim
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, transformer_embed_dim)
        
        # Project input features to transformer dimension
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, transformer_embed_dim),
            nn.LayerNorm(transformer_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_embed_dim, transformer_embed_dim)
        )
        
        # Positional encoding (optional)
        if use_pos_enc:
            self.pos_enc = nn.Linear(3, transformer_embed_dim)
        else:
            self.pos_enc = None
        
        # Transformer decoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=transformer_embed_dim,
                n_head=transformer_n_head,
                hidden_dim=transformer_hidden_dim
            )
            for _ in range(n_transformer_layer)
        ])
        
        # Mask prediction head
        self.mask_embed = nn.Sequential(
            nn.Linear(transformer_embed_dim, transformer_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_embed_dim, transformer_embed_dim)
        )
    
    def forward(self, point_features, point_coords=None, point=None):
        """
        Forward pass with batch-wise processing.
        
        Args:
            point_features: [N_total, in_channels] features
            point_coords: [N_total, 3] point coordinates (optional, for positional encoding)
            point: Point object (must contain .batch for batch indices)
        
        Returns:
            masks: [K, N_total] segmentation mask logits (concatenated from all batches)
        """
        features = point_features  # [N_total, in_channels]
        coords = point_coords
        
        # Get batch indices from point object
        batch = point.batch  # [N_total]
        unique_batches = torch.unique(batch)
        
        # Project all features at once for efficiency
        proj_features = self.input_proj(features)  # [N_total, transformer_embed_dim]
        
        # Positional encoding (if enabled)
        pos_enc = None
        if self.pos_enc is not None and coords is not None:
            pos_enc = self.pos_enc(coords)  # [N_total, D]
        
        # Process each batch separately
        mask_logits_list = []
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_features = proj_features[batch_mask]  # [N_b, D]
            N_b = batch_features.shape[0]
            
            # Get query embeddings for this batch
            queries = self.query_embed.weight.unsqueeze(1)  # [K, 1, D]
            queries = queries.expand(-1, N_b, -1)  # [K, N_b, D]
            
            # Get positional encoding for this batch
            batch_pos_enc = None
            if pos_enc is not None:
                batch_pos_enc = pos_enc[batch_mask]  # [N_b, D]
            
            # Apply transformer decoder layers
            for layer in self.transformer_layers:
                queries = layer(queries, batch_features, batch_pos_enc)
            
            # Generate mask embeddings
            mask_embed = self.mask_embed(queries)  # [K, N_b, D]
            
            # Compute mask logits
            mask_embed_norm = F.normalize(mask_embed, dim=-1)  # [K, N_b, D]
            features_norm = F.normalize(batch_features, dim=-1)  # [N_b, D]
            
            batch_mask_logits = torch.einsum('knd,nd->kn', mask_embed_norm, features_norm)  # [K, N_b]
            batch_mask_logits = batch_mask_logits / 0.07
            
            mask_logits_list.append(batch_mask_logits)
        
        # Concatenate all batch results: [K, N_total]
        mask_logits = torch.cat(mask_logits_list, dim=1)
        
        return mask_logits
    

