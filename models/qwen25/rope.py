import torch
from transformers import Qwen2Config

def _compute_rope_params(base, rope_dim, max_position_embeddings):

    # frequency 1 / 10000^{2i/d}
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2) / rope_dim))     # dim // 2  

    position_ids_expanded = torch.arange(0, max_position_embeddings).reshape(1, max_position_embeddings)     # (1, T)
    inv_freq_expanded = inv_freq.reshape(-1, 1)          # (dim//2, 1)

    freqs = (inv_freq_expanded @ position_ids_expanded.float()).transpose(0, 1)

    emb = torch.cat((freqs, freqs), dim=-1)    # (T, dim)                # RoPE 可以是交错的，也可以是对半分的

    cos = emb.cos()
    sin = emb.sin()

    return cos, sin

def rotate_half(x):
    '''
     q1  q2  q3 q4 q5 q6
    --->
    -q4 -q5 -q5 q1 q2 q3
    '''
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2: ]
    return torch.cat((-x2, x1), -1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + rotate_half(q) * sin
    k_embed = (k * cos) + rotate_half(k) * sin

    return q_embed, k_embed

