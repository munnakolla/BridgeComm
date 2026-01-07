"""
Vision Transformer Architecture for SignVLM
============================================

Based on the original ViT architecture with CLIP-compatible modifications.
Includes:
- LayerNorm with fp16 support
- QuickGELU activation
- PatchEmbed2D for image patching
- VisionTransformer2D backbone

Source: https://github.com/Hamzah-Luqman/signVLM
"""

from collections import OrderedDict
import numpy as np
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    """
    Fast GELU approximation from CLIP.
    Uses sigmoid(1.702 * x) * x instead of exact GELU.
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16.
    Casts to float32 for the forward pass, then back to original dtype.
    """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Attention(nn.Module):
    """
    Multi-head self-attention module.
    
    Args:
        feature_dim: Input/output feature dimension
        num_heads: Number of attention heads
        qkv_dim: Dimension for Q, K, V projections
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        qkv_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.in_proj = nn.Linear(feature_dim, qkv_dim * 3)
        self.out_proj = nn.Linear(qkv_dim, feature_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_qk: bool = False,
    ):
        N, L, C = x.size()
        qkv = self.in_proj(x).view(N, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(N, L, -1)
        x = self.out_proj(x)
        
        if return_qk:
            return x, q.flatten(2), k.flatten(2)
        return x


class PatchEmbed2D(nn.Module):
    """
    2D Image to Patch Embedding.
    
    Converts an image into a sequence of flattened patch embeddings.
    
    Args:
        patch_size: Size of each patch (height, width)
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1], embed_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Patch embeddings of shape (N, num_patches, embed_dim)
        """
        N, C, H, W = x.size()
        pH, pW = self.patch_size
        
        # Unfold into patches
        x = x.unfold(2, pH, pH).unfold(3, pW, pW)  # N, C, nH, nW, pH, pW
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # N, nH, nW, C, pH, pW
        x = x.view(N, -1, C * pH * pW)  # N, num_patches, patch_dim
        
        x = self.proj(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with pre-norm.
    
    Args:
        feature_dim: Feature dimension
        qkv_dim: QKV projection dimension
        num_heads: Number of attention heads
        mlp_factor: MLP expansion factor
    """
    
    def __init__(
        self,
        feature_dim: int,
        qkv_dim: int,
        num_heads: int,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
    ):
        super().__init__()
        
        self.ln_1 = LayerNorm(feature_dim)
        self.attn = Attention(feature_dim, num_heads, qkv_dim)
        
        self.ln_2 = LayerNorm(feature_dim)
        mlp_dim = int(feature_dim * mlp_factor)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, mlp_dim)),
            ("gelu", act()),
            ("c_proj", nn.Linear(mlp_dim, feature_dim)),
        ]))
    
    def forward(
        self,
        x: torch.Tensor,
        return_qk: bool = False,
    ):
        if return_qk:
            attn_out, q, k = self.attn(self.ln_1(x), return_qk=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, q, k
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with cross-attention.
    
    Args:
        feature_dim: Feature dimension
        qkv_dim: QKV projection dimension
        num_heads: Number of attention heads
        mlp_factor: MLP expansion factor
        mlp_dropout: Dropout rate for MLP
    """
    
    def __init__(
        self,
        feature_dim: int,
        qkv_dim: int,
        num_heads: int,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.ln_1 = LayerNorm(feature_dim)
        self.self_attn = Attention(feature_dim, num_heads, qkv_dim)
        
        self.ln_2 = LayerNorm(feature_dim)
        self.cross_attn = Attention(feature_dim, num_heads, qkv_dim)
        
        self.ln_3 = LayerNorm(feature_dim)
        mlp_dim = int(feature_dim * mlp_factor)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, mlp_dim)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(mlp_dropout)),
            ("c_proj", nn.Linear(mlp_dim, feature_dim)),
        ]))
    
    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor (N, L_q, C)
            mem: Memory tensor (N, L_m, C)
            
        Returns:
            Output tensor (N, L_q, C)
        """
        # Self-attention
        x = x + self.self_attn(self.ln_1(x))
        
        # Cross-attention
        x_norm = self.ln_2(x)
        N, L_q, C = x_norm.size()
        _, L_m, _ = mem.size()
        
        # Concatenate for cross-attention
        xm = torch.cat([x_norm, mem], dim=1)
        xm_attn = self.cross_attn(xm)
        x = x + xm_attn[:, :L_q, :]
        
        # MLP
        x = x + self.mlp(self.ln_3(x))
        
        return x


class VisionTransformer2D(nn.Module):
    """
    2D Vision Transformer backbone.
    
    Compatible with CLIP ViT weights. Processes single frames.
    
    Args:
        feature_dim: Transformer feature dimension
        input_size: Input image size (H, W)
        patch_size: Patch size (pH, pW)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_factor: MLP expansion factor
        act: Activation function class
        return_all_features: Return features from all layers
        ln_pre: Apply LayerNorm before transformer (CLIP style)
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        input_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        return_all_features: bool = False,
        ln_pre: bool = False,
    ):
        super().__init__()
        
        self.return_all_features = return_all_features
        self.num_layers = num_layers
        
        self.patch_embed = PatchEmbed2D(patch_size, 3, feature_dim)
        
        num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, feature_dim))
        
        self.ln_pre = LayerNorm(feature_dim) if ln_pre else nn.Identity()
        
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(feature_dim, feature_dim, num_heads, mlp_factor, act)
            for _ in range(num_layers)
        ])
        
        self.ln_post = LayerNorm(feature_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input images of shape (N, C, H, W)
            
        Returns:
            List of layer outputs if return_all_features, else final output
        """
        N = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pre-normalization (CLIP style)
        x = self.ln_pre(x)
        
        # Transformer blocks
        all_features = []
        for i, block in enumerate(self.blocks):
            if self.return_all_features:
                x, q, k = block(x, return_qk=True)
                # Exclude CLS token from features
                all_features.append({
                    'out': x[:, 1:, :],  # (N, num_patches, C)
                    'q': q[:, :, 1:],    # Query without CLS
                    'k': k[:, :, 1:],    # Key without CLS
                })
            else:
                x = block(x)
        
        x = self.ln_post(x)
        
        if self.return_all_features:
            return all_features
        else:
            return x[:, 0, :]  # Return CLS token


def model_to_fp16(model: VisionTransformer2D):
    """Convert model to fp16 while keeping LayerNorm in fp32."""
    def _convert_to_fp16(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()
        elif isinstance(module, nn.MultiheadAttention):
            for attr in ['in_proj_weight', 'out_proj.weight', 'bias_k', 'bias_v']:
                tensor = getattr(module, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()
    
    model.apply(_convert_to_fp16)


# ViT Presets for different model sizes
vit_presets = {
    'ViT-B/16': dict(
        feature_dim=768,
        input_size=(224, 224),
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
        ln_pre=False,
    ),
    'ViT-B/16-lnpre': dict(
        feature_dim=768,
        input_size=(224, 224),
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
        ln_pre=True,
    ),
    'ViT-L/14': dict(
        feature_dim=1024,
        input_size=(224, 224),
        patch_size=(14, 14),
        num_heads=16,
        num_layers=24,
        mlp_factor=4.0,
        ln_pre=False,
    ),
    'ViT-L/14-lnpre': dict(
        feature_dim=1024,
        input_size=(224, 224),
        patch_size=(14, 14),
        num_heads=16,
        num_layers=24,
        mlp_factor=4.0,
        ln_pre=True,
    ),
}
