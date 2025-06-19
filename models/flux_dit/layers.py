import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from dataclasses import dataclass
from typing import Tuple, List
from .rope_attention import RopeSelfAttention, rope_attention, EmbedND
from .norm import QKNorm

@dataclass
class ModulationOutput:
    shift: Tensor
    scale: Tensor
    gate: Tensor

class Modulation:
    def __init__(self, dim: int, is_double: bool):
        self.is_double = is_double
        self.multiplier = 6 if is_double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)
        
    def forward(self, vec: Tensor) -> Tuple[ModulationOutput, ModulationOutput] | Tuple[ModulationOutput, None]:
        vec = F.silu(vec)
        out = self.lin(vec)[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOutput(*out[: 3]),
            ModulationOutput(*out[3: ]) if self.is_double else None
        )


class MLP(nn.Module):
    def __init__(self, in_out_dim: int, hidden_dim: int, bias=True, act_func='silu'):
        super().__init__()
        self.in_layer = nn.Linear(in_out_dim, hidden_dim, bias)

        if act_func.lower() == 'silu':
            self.act = nn.SiLU()
        elif act_func.lower() == 'gelu':
            self.act = nn.GELU(approximate='tanh')
        
        self.out_layer = nn.Linear(hidden_dim, in_out_dim, bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layer(x)
        x = self.act(x)
        x = self.out_layer(x)
        return x


class DoubleStreamBlock(nn.Module):
    '''
    aka MM-DiT in Stable Diffusion 3 
    '''
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = Modulation(hidden_size, is_double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = RopeSelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = MLP(in_out_dim=hidden_size, hidden_dim=mlp_hidden_dim, bias=True, act_func='gelu')

        self.txt_mod = Modulation(hidden_size, is_double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = RopeSelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = MLP(in_out_dim=hidden_size, hidden_dim=mlp_hidden_dim, bias=True, act_func='gelu')

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> Tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + txt_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)         # 这里 Attention class 为啥要这样实现？？？
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.qk_norm(img_q, img_k, img_v)

        # prepare text for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)         
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.qk_norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = rope_attention(q, k, v, pe)
        txt_attn = attn[:, : txt.shape[1]]
        img_attn = attn[:, txt.shape[1]: ]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(1 + img_mod2.scale * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(1 + txt_mod2.scale * self.txt_norm2(txt) + txt_mod2.shift)

        return img, txt

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: float | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.qk_norm = QKNorm(head_dim)

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.mlp_act = nn.GELU(approximate='tanh')
        self.modulation = Modulation(hidden_size, is_double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor):
        # x 是 concat(txt, img)
        mod, _ = self.modulation(vec)

        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        qkv, mlp = torch.split(self.linear1(x_mod, [3 * self.hidden_size, self.mlp_hidden_dim]), dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.qk_norm(q, k, v)

        # compute attenion
        attn = rope_attention(q, k, v, pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat(attn, self.mlp_act(mlp), 2))

        return x + mod.gate * output

class LasyLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size ** 2 * patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2, bias=True)
        )

    def foward(self, x: Tensor, vec: Tensor):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm(x) + shift[:, None, :]
        x = self.linear(x)
        return x
