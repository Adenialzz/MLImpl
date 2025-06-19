import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import math
from .norm import QKNorm

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
    
    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim = -3
        )
        return emb.unsqueeze(dim=1)



def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0

    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim

    omega = 1.0 / (theta ** scale)

    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)

    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def apply_rope(xq, xk, freq_cis) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[: -1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[: -1], -1, 1, 2)

    xq_out = freq_cis[..., 0] * xq_[..., 0] + freq_cis[..., 1] * xq_[..., 1]
    xk_out = freq_cis[..., 0] * xk_[..., 0] + freq_cis[..., 1] * xk_[..., 1]

    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

def rope_attention(q, k, v, pe):  # pe 就是 freq_cis，有 EmbedND 产生
    q, k = apply_rope(q, k, pe)

    x = F.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


class RopeSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.qk_norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q, k = self.qk_norm(q, k, v)

        x = rope_attention(q, k, v, pe)
        x = self.proj(x)

        return x


