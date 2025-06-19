import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import List
import math
from .layers import DoubleStreamBlock, SingleStreamBlock, LasyLayer, MLP
from .rope_attention import EmbedND

@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dims: List[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


def timestep_embedding(t: Tensor, dim, max_period: int = 10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """

    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32 / half)).to(t.device)

    args = t[:, None].float() * freqs[None]  # tensor[xxx, None, xxx] 相当于 tensor.unsqueeze(dim=)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)

    return embedding



class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params

        assert params.hidden_size % params.num_heads == 0, f"hidden size {params.hidden_size} must be divisible by num heads {params.num_heads}"

        pe_dim = params.hidden_size // params.num_heads  # head dim

        assert len(params.axes_dims) == pe_dim, f"length of axes dims {params.axes_dims} must equal to positional encoding dim {pe_dim}"

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dims)

        self.img_in = nn.Linear(params.in_channels, params.hidden_size, bias=True)
        self.time_in = MLP(in_out_dim=256, hidden_dim=params.hidden_size, act_func='silu')
        self.vector_in = MLP(in_out_dim=params.vec_in_dim, hidden_dim=params.hidden_size, act_func='silu')
        
        if params.guidance_embed:
            self.guidance_in = MLP(in_out_dim=256, hidden_dim=params.hidden_size, act_func='silu')
        else:
            self.guidance_in = nn.Identity()

        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size=params.hidden_size,
                    num_heads=params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=params.hidden_size,
                    num_heads=params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qk_scale=None
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.last_layer = LasyLayer(params.hidden_size, 1, params.out_channels)  # patch size = 1 ???

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor, # t5 text embedding
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,  # clip text embedding。与 guidance 和 timestep 经编码后一起组成 vec，通过 modulation 用于提供条件信息
        guidance: Tensor | None = None
    ):
        assert img.ndim == 3 and txt.ndim == 3, f"input image and text tensors must have 3 dimensions"

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))

        if self.params.guidance_embed:
            assert guidance is not None, "didn't get guidance for guidance distilled model."

            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))  

        vec = vec + self.vector_in(y)
        
        # vec 是什么？：用正弦编码分别编码 guidance (int) 和 timestep (int)，还有 clip text embedding，三者加起来就是 vec 

        txt = self.txt_in(txt)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)


        img = torch.cat([img, txt], dim=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)

        img = self.last_layer(img, vec)   # (N, T, patch_size ** 2 * out_channels)
        return img
