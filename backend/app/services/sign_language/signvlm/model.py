"""
EVL Transformer Model for SignVLM
=================================

Main model architecture combining:
- CLIP ViT backbone (frozen fp16)
- Temporal cross-attention decoder
- Classification head

The model processes video frames through:
1. CLIP backbone: Extract spatial features from each frame
2. Temporal convolution: Model local temporal patterns
3. Cross-attention decoder: Aggregate temporal information
4. Classification head: Predict sign language class

Source: https://github.com/Hamzah-Luqman/signVLM
"""

from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import (
    QuickGELU, Attention, LayerNorm,
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)
from .weight_loaders import weight_loader_fn_dict


class TemporalCrossAttention(nn.Module):
    """
    Temporal cross-attention module.
    
    Computes attention between query and key features from different frames
    to model temporal relationships.
    
    Args:
        spatial_size: Spatial dimensions (H, W) of feature maps
        feature_dim: Feature dimension
    """
    
    def __init__(
        self,
        spatial_size: Tuple[int, int],
        feature_dim: int,
    ):
        super().__init__()
        
        self.spatial_size = spatial_size
        num_patches = spatial_size[0] * spatial_size[1]
        
        # Learnable spatial position embeddings for cross-attention
        self.pos_embed_q = nn.Parameter(torch.zeros(1, 1, num_patches, feature_dim))
        self.pos_embed_k = nn.Parameter(torch.zeros(1, 1, num_patches, feature_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.pos_embed_q, std=0.02)
        nn.init.normal_(self.pos_embed_k, std=0.02)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal cross-attention.
        
        Args:
            q: Query features (N, T, num_heads * head_dim, L)
            k: Key features (N, T, num_heads * head_dim, L)
            
        Returns:
            Cross-attention output (N, T, L, C)
        """
        N, T, D, L = q.size()
        
        # Reshape for attention
        q = q.permute(0, 1, 3, 2)  # N, T, L, D
        k = k.permute(0, 1, 3, 2)  # N, T, L, D
        
        # Add position embeddings
        q = q + self.pos_embed_q
        k = k + self.pos_embed_k
        
        # Compute temporal attention
        # Each position attends to the same position in other frames
        attn = torch.einsum('ntlc,nslc->ntsl', q, k)  # N, T, T, L
        attn = attn / (D ** 0.5)
        attn = F.softmax(attn, dim=2)
        
        # Apply attention to values (using k as v for simplicity)
        out = torch.einsum('ntsl,nslc->ntlc', attn, k)  # N, T, L, D
        
        return out


class EVLDecoder(nn.Module):
    """
    Efficient Video Learning Decoder.
    
    Processes frame features through temporal modeling and cross-attention
    to produce a video-level representation.
    
    Args:
        num_frames: Number of input frames
        spatial_size: Spatial size of feature maps
        num_layers: Number of decoder layers
        in_feature_dim: Input feature dimension
        qkv_dim: QKV projection dimension
        num_heads: Number of attention heads
        mlp_factor: MLP expansion factor
        enable_temporal_conv: Use temporal convolution
        enable_temporal_pos_embed: Use temporal position embeddings
        enable_temporal_cross_attention: Use temporal cross-attention
        mlp_dropout: Dropout rate for MLP
    """
    
    def __init__(
        self,
        num_frames: int = 8,
        spatial_size: Tuple[int, int] = (14, 14),
        num_layers: int = 4,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()
        
        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout
            ) for _ in range(num_layers)
        ])
        
        # Temporal convolution for local temporal modeling
        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList([
                nn.Conv1d(
                    in_feature_dim, in_feature_dim,
                    kernel_size=3, stride=1, padding=1,
                    groups=in_feature_dim  # Depthwise convolution
                ) for _ in range(num_layers)
            ])
        
        # Temporal position embeddings
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList([
                nn.Parameter(torch.zeros(num_frames, in_feature_dim))
                for _ in range(num_layers)
            ])
        
        # Temporal cross-attention
        if enable_temporal_cross_attention:
            self.cross_attention = nn.ModuleList([
                TemporalCrossAttention(spatial_size, in_feature_dim)
                for _ in range(num_layers)
            ])
        
        # Class token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(in_feature_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        if self.enable_temporal_pos_embed:
            for pos_embed in self.temporal_pos_embed:
                nn.init.normal_(pos_embed, std=0.02)
    
    def forward(self, in_features: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Process frame features and produce video representation.
        
        Args:
            in_features: List of dicts with 'out', 'q', 'k' tensors
                - out: Frame features (N, T, L, C)
                - q: Query features (N, T, D, L)
                - k: Key features (N, T, D, L)
                
        Returns:
            Video representation (N, 1, C)
        """
        N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        
        # Initialize with class token
        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)  # N, 1, C
        
        for i in range(self.num_layers):
            frame_features = in_features[i]['out']  # N, T, L, C
            
            # Apply temporal convolution
            if self.enable_temporal_conv:
                feat = in_features[i]['out']  # N, T, L, C
                feat = feat.permute(0, 2, 3, 1).contiguous().flatten(0, 1)  # N*L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C, T).permute(0, 3, 1, 2).contiguous()  # N, T, L, C
                frame_features = frame_features + feat
            
            # Add temporal position embeddings
            if self.enable_temporal_pos_embed:
                frame_features = frame_features + self.temporal_pos_embed[i].view(1, T, 1, C)
            
            # Apply temporal cross-attention
            if self.enable_temporal_cross_attention:
                cross_attn_out = self.cross_attention[i](
                    in_features[i]['q'], in_features[i]['k']
                )
                frame_features = frame_features + cross_attn_out
            
            # Flatten spatial and temporal dimensions
            frame_features = frame_features.flatten(1, 2)  # N, T * L, C
            
            # Decoder layer
            x = self.decoder_layers[i](x, frame_features)
        
        return x


class EVLTransformer(nn.Module):
    """
    Main SignVLM model: EVL Transformer for video classification.
    
    Combines a CLIP ViT backbone with a temporal decoder for
    video-based sign language recognition.
    
    Args:
        num_frames: Number of input video frames
        backbone_name: ViT backbone variant (e.g., 'ViT-L/14-lnpre')
        backbone_type: Type of pretrained weights ('clip', 'mae', 'timm')
        backbone_path: Path to pretrained backbone weights
        backbone_mode: How to use backbone ('freeze_fp16', 'freeze_fp32', 'finetune')
        decoder_num_layers: Number of decoder layers
        decoder_qkv_dim: Decoder QKV dimension
        decoder_num_heads: Decoder attention heads
        decoder_mlp_factor: Decoder MLP expansion factor
        num_classes: Number of output classes
        enable_temporal_conv: Use temporal convolution
        enable_temporal_pos_embed: Use temporal position embeddings
        enable_temporal_cross_attention: Use temporal cross-attention
        cls_dropout: Dropout rate for classification head
        decoder_mlp_dropout: Dropout rate for decoder MLP
    """
    
    def __init__(
        self,
        num_frames: int = 8,
        backbone_name: str = 'ViT-B/16',
        backbone_type: str = 'clip',
        backbone_path: str = '',
        backbone_mode: str = 'freeze_fp16',
        decoder_num_layers: int = 4,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_mlp_factor: float = 4.0,
        num_classes: int = 400,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        cls_dropout: float = 0.5,
        decoder_mlp_dropout: float = 0.5,
    ):
        super().__init__()
        
        self.decoder_num_layers = decoder_num_layers
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        print(f'[SignVLM] Creating backbone: {backbone_name}')
        backbone_config = self._create_backbone(
            backbone_name, backbone_type, backbone_path, backbone_mode
        )
        
        backbone_feature_dim = backbone_config['feature_dim']
        backbone_spatial_size = tuple(
            x // y for x, y in zip(
                backbone_config['input_size'],
                backbone_config['patch_size']
            )
        )
        
        print(f'[SignVLM] Backbone feature dim: {backbone_feature_dim}')
        print(f'[SignVLM] Backbone spatial size: {backbone_spatial_size}')
        
        # Create decoder
        self.decoder = EVLDecoder(
            num_frames=num_frames,
            spatial_size=backbone_spatial_size,
            num_layers=decoder_num_layers,
            in_feature_dim=backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            enable_temporal_conv=enable_temporal_conv,
            enable_temporal_pos_embed=enable_temporal_pos_embed,
            enable_temporal_cross_attention=enable_temporal_cross_attention,
            mlp_dropout=decoder_mlp_dropout,
        )
        
        # Classification head
        self.proj = nn.Sequential(
            LayerNorm(backbone_feature_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(backbone_feature_dim, num_classes),
        )
        
        print(f'[SignVLM] Model initialized with {num_classes} classes')
    
    def _create_backbone(
        self,
        backbone_name: str,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
    ) -> dict:
        """Create and initialize the backbone."""
        
        # Load pretrained weights
        if backbone_path and backbone_type in weight_loader_fn_dict:
            weight_loader_fn = weight_loader_fn_dict[backbone_type]
            state_dict = weight_loader_fn(backbone_path)
        else:
            state_dict = None
        
        # Create backbone
        backbone = VisionTransformer2D(
            return_all_features=True,
            **vit_presets[backbone_name]
        )
        
        # Load weights
        if state_dict is not None:
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            if missing:
                print(f'[SignVLM] Missing keys: {missing[:5]}...')
            if unexpected:
                print(f'[SignVLM] Unexpected keys: {unexpected[:5]}...')
        
        # Configure backbone mode
        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']
        
        if backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval()
            backbone.requires_grad_(False)
            if backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)
            # Store as list to avoid parameter registration
            self.backbone = [backbone]
        
        return vit_presets[backbone_name]
    
    def _get_backbone(self, x: torch.Tensor):
        """Get backbone, moving to correct device if needed."""
        if isinstance(self.backbone, list):
            # Frozen backbone stored as list
            self.backbone[0] = self.backbone[0].to(x.device)
            return self.backbone[0]
        else:
            # Trainable backbone
            return self.backbone
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for video classification.
        
        Args:
            x: Input video tensor (N, C, T, H, W)
               - N: Batch size
               - C: Channels (3)
               - T: Number of frames
               - H, W: Height and width (224)
               
        Returns:
            Logits tensor (N, num_classes)
        """
        backbone = self._get_backbone(x)
        
        B, C, T, H, W = x.size()
        
        # Reshape to process all frames through backbone
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # B*T, C, H, W
        
        # Get features from last decoder_num_layers layers
        features = backbone(x)[-self.decoder_num_layers:]
        
        # Reshape features back to video format
        features = [
            dict(
                (k, v.float().view(B, T, *v.size()[1:]))
                for k, v in layer_feat.items()
            )
            for layer_feat in features
        ]
        
        # Decode to get video representation
        x = self.decoder(features)  # B, 1, C
        
        # Classify
        x = self.proj(x[:, 0, :])  # B, num_classes
        
        return x
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            strict: Whether to require exact key matching
        """
        print(f'[SignVLM] Loading checkpoint from {checkpoint_path}')
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle distributed training checkpoints
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        # Remove 'module.' prefix from DDP training
        state_dict = {
            k.replace('module.', ''): v
            for k, v in state_dict.items()
        }
        
        # Load state dict (skip backbone if frozen)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f'[SignVLM] Missing keys: {len(missing)}')
        if unexpected:
            print(f'[SignVLM] Unexpected keys: {len(unexpected)}')
        
        print(f'[SignVLM] Checkpoint loaded successfully')
