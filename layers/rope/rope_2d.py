import torch
import torch.nn as nn
import math


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding (RoPE) implementation.
    Applies rotary embeddings to 2D spatial data (e.g., images, feature maps).
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Generate frequency bands for both dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len_h=None, seq_len_w=None):
        """
        Apply 2D rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len_h, seq_len_w, dim)
            seq_len_h: Height sequence length (optional, inferred from x if not provided)
            seq_len_w: Width sequence length (optional, inferred from x if not provided)
            
        Returns:
            Tensor with rotary position embeddings applied
        """
        if seq_len_h is None:
            seq_len_h = x.shape[1]
        if seq_len_w is None:
            seq_len_w = x.shape[2]
            
        device = x.device
        dtype = x.dtype
        
        # Create position indices for both dimensions
        pos_h = torch.arange(seq_len_h, device=device, dtype=dtype)
        pos_w = torch.arange(seq_len_w, device=device, dtype=dtype)
        
        # Compute frequencies for both dimensions
        freqs_h = torch.outer(pos_h, self.inv_freq)
        freqs_w = torch.outer(pos_w, self.inv_freq)
        
        # Create 2D position embeddings
        emb_h = torch.cat((freqs_h, freqs_h), dim=-1)
        emb_w = torch.cat((freqs_w, freqs_w), dim=-1)
        
        # Apply sin/cos transformations
        cos_h = emb_h.cos()
        sin_h = emb_h.sin()
        cos_w = emb_w.cos()
        sin_w = emb_w.sin()
        
        # Reshape for broadcasting
        cos_h = cos_h.unsqueeze(1).unsqueeze(0)  # (1, seq_len_h, 1, dim)
        sin_h = sin_h.unsqueeze(1).unsqueeze(0)  # (1, seq_len_h, 1, dim)
        cos_w = cos_w.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_w, dim)
        sin_w = cos_w.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_w, dim)
        
        # Split input into even and odd dimensions
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # Apply 2D rotary transformations
        # For height dimension
        x_h_rotated = torch.cat([
            x_even * cos_h - x_odd * sin_h,
            x_even * sin_h + x_odd * cos_h
        ], dim=-1)
        
        # For width dimension
        x_w_rotated = torch.cat([
            x_h_rotated[..., ::2] * cos_w - x_h_rotated[..., 1::2] * sin_w,
            x_h_rotated[..., ::2] * sin_w + x_h_rotated[..., 1::2] * cos_w
        ], dim=-1)
        
        return x_w_rotated



def apply_rope_2d(x, dim, max_position_embeddings=2048, base=10000, seq_len_h=None, seq_len_w=None):
    """
    Functional interface for applying 2D RoPE to a tensor.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len_h, seq_len_w, dim)
        dim: Embedding dimension
        max_position_embeddings: Maximum position embeddings
        base: Base for frequency computation
        seq_len_h: Height sequence length
        seq_len_w: Width sequence length
        
    Returns:
        Tensor with 2D RoPE applied
    """
    rope = RoPE2D(dim, max_position_embeddings, base)
    return rope(x, seq_len_h, seq_len_w)


from transformers import Qwen2VLForConditionalGeneration
