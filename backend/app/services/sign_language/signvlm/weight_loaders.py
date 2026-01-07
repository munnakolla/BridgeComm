"""
Weight Loaders for SignVLM
==========================

Functions to load pretrained weights from different sources:
- CLIP: OpenAI CLIP ViT weights
- MAE: Masked Autoencoder weights

Source: https://github.com/Hamzah-Luqman/signVLM
"""

from typing import Dict
import torch


def load_weights_clip(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Load CLIP ViT weights and convert to our format.
    
    CLIP uses a JIT-compiled model, so we need to extract the visual encoder
    and remap the weight names.
    
    Args:
        load_path: Path to CLIP .pt file (e.g., ViT-L-14.pt)
        
    Returns:
        State dict compatible with VisionTransformer2D
    """
    # Load CLIP model
    clip_model = torch.jit.load(load_path, map_location='cpu')
    clip_model = clip_model.visual
    src_state_dict = clip_model.state_dict()
    src_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())

    dst_state_dict = {}
    
    # Map special tokens and embeddings
    dst_state_dict['cls_token'] = src_state_dict['class_embedding'].unsqueeze(0).unsqueeze(0)
    dst_state_dict['pos_embed'] = src_state_dict['positional_embedding'].unsqueeze(0)
    
    # Patch embedding
    dst_state_dict['patch_embed.proj.weight'] = src_state_dict['conv1.weight'].flatten(1)
    dst_state_dict['patch_embed.proj.bias'] = torch.zeros([src_state_dict['conv1.weight'].size(0)])
    
    # Pre-normalization (if present in CLIP)
    if 'ln_pre.weight' in src_state_dict:
        dst_state_dict['ln_pre.weight'] = src_state_dict['ln_pre.weight']
        dst_state_dict['ln_pre.bias'] = src_state_dict['ln_pre.bias']

    # Transformer blocks
    block_idx = 0
    while True:
        src_prefix = 'transformer.resblocks.%d.' % block_idx
        dst_prefix = 'blocks.%d.' % block_idx
        
        # Check if this block exists
        if src_prefix + 'ln_1.weight' not in src_state_dict:
            break
        
        # Layer norms
        dst_state_dict[dst_prefix + 'ln_1.weight'] = src_state_dict[src_prefix + 'ln_1.weight']
        dst_state_dict[dst_prefix + 'ln_1.bias'] = src_state_dict[src_prefix + 'ln_1.bias']
        dst_state_dict[dst_prefix + 'ln_2.weight'] = src_state_dict[src_prefix + 'ln_2.weight']
        dst_state_dict[dst_prefix + 'ln_2.bias'] = src_state_dict[src_prefix + 'ln_2.bias']
        
        # Attention
        dst_state_dict[dst_prefix + 'attn.in_proj.weight'] = src_state_dict[src_prefix + 'attn.in_proj_weight']
        dst_state_dict[dst_prefix + 'attn.in_proj.bias'] = src_state_dict[src_prefix + 'attn.in_proj_bias']
        dst_state_dict[dst_prefix + 'attn.out_proj.weight'] = src_state_dict[src_prefix + 'attn.out_proj.weight']
        dst_state_dict[dst_prefix + 'attn.out_proj.bias'] = src_state_dict[src_prefix + 'attn.out_proj.bias']
        
        # MLP
        dst_state_dict[dst_prefix + 'mlp.c_fc.weight'] = src_state_dict[src_prefix + 'mlp.c_fc.weight']
        dst_state_dict[dst_prefix + 'mlp.c_fc.bias'] = src_state_dict[src_prefix + 'mlp.c_fc.bias']
        dst_state_dict[dst_prefix + 'mlp.c_proj.weight'] = src_state_dict[src_prefix + 'mlp.c_proj.weight']
        dst_state_dict[dst_prefix + 'mlp.c_proj.bias'] = src_state_dict[src_prefix + 'mlp.c_proj.bias']
        
        block_idx += 1
    
    # Post-normalization
    if 'ln_post.weight' in src_state_dict:
        dst_state_dict['ln_post.weight'] = src_state_dict['ln_post.weight']
        dst_state_dict['ln_post.bias'] = src_state_dict['ln_post.bias']
    
    return dst_state_dict


def load_weights_mae(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Load MAE (Masked Autoencoder) weights.
    
    MAE weights are typically closer to our format but may need minor remapping.
    
    Args:
        load_path: Path to MAE checkpoint file
        
    Returns:
        State dict compatible with VisionTransformer2D
    """
    ckpt = torch.load(load_path, map_location='cpu')
    
    # MAE checkpoints may have 'model' or 'state_dict' key
    if 'model' in ckpt:
        src_state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        src_state_dict = ckpt['state_dict']
    else:
        src_state_dict = ckpt
    
    # Convert to float32
    dst_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())
    
    return dst_state_dict


def load_weights_timm(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Load timm (PyTorch Image Models) weights.
    
    Args:
        load_path: Path to timm checkpoint file
        
    Returns:
        State dict compatible with VisionTransformer2D
    """
    ckpt = torch.load(load_path, map_location='cpu')
    
    if 'model' in ckpt:
        src_state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        src_state_dict = ckpt['state_dict']
    else:
        src_state_dict = ckpt
    
    dst_state_dict = {}
    
    # Map from timm naming to our naming
    for k, v in src_state_dict.items():
        # Skip head/classifier weights
        if 'head' in k or 'fc_norm' in k:
            continue
        
        # Rename keys
        new_k = k
        new_k = new_k.replace('patch_embed.proj', 'patch_embed.proj')
        new_k = new_k.replace('norm.', 'ln_post.')
        new_k = new_k.replace('.norm1.', '.ln_1.')
        new_k = new_k.replace('.norm2.', '.ln_2.')
        new_k = new_k.replace('.mlp.fc1.', '.mlp.c_fc.')
        new_k = new_k.replace('.mlp.fc2.', '.mlp.c_proj.')
        
        dst_state_dict[new_k] = v.float()
    
    return dst_state_dict


# Registry of weight loaders
weight_loader_fn_dict = {
    'clip': load_weights_clip,
    'mae': load_weights_mae,
    'timm': load_weights_timm,
}
