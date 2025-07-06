import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from rope import apply_rotary_pos_emb

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdims=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(x_dtype)

class SwiGLU(nn.Module):
    def __init__(self, dim, immediate_dim, bias=False):
        super().__init__()
        self.up_proj = nn.Linear(dim, immediate_dim, bias=bias)
        self.down_proj = nn.Linear(immediate_dim, dim, bias=bias)
        self.gate_proj = nn.Linear(dim, immediate_dim, bias=bias)

    def forward(self, x):
        x, gate = self.up_proj(x), self.gate_proj(x)
        x = F.silu(gate) * x
        x = self.down_proj(x)
        return x


# https://github.com/huggingface/transformers/blob/ca7e1a3756c022bf31429c452b2f313f043f32de/src/transformers/models/qwen2/modeling_qwen2.py#L86C1-L97C1
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


    
class CausalGroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_key_value_heads, proj_bias=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = hidden_dim // num_attention_heads

        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_key_value_heads * self.head_dim)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=proj_bias)

    def forward(self, x, cos=None, sin=None):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)     # (B, N, T, H)
        k = self.k_proj(x).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)     # (B, N_G, T, H)
        v = self.v_proj(x).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = q @ repeat_kv(k, self.num_key_value_groups).transpose(-1, -2) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T).view(1, 1, T, T)).to(attn.device)    #  apply causal mask
        attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))     

        attn = F.softmax(attn, dim=-1)

        o = attn @ repeat_kv(v, self.num_key_value_groups)
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        o = self.o_proj(o)

        return o



class Block(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_key_value_heads, immediate_dim):
        super().__init__()

        self.self_attn = CausalGroupQueryAttention(hidden_dim, num_attention_heads, num_key_value_heads)
        self.mlp = SwiGLU(hidden_dim, immediate_dim)
        self.input_layernorm = RMSNorm(hidden_dim)
        self.post_attention_layernorm = RMSNorm(hidden_dim)

    def forward(self, x, cos=None, sin=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)  # pre norm
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x

