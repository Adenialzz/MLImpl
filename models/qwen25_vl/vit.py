import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

class PatchEmbed(nn.Module):
    def __init__(self, patch_size = 14, temporal_patch_size = 2, in_channels = 3, embed_dim = 1152):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels=in_channels, out_channels=embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype

        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )

        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
        hidden_states = hidden_states.view(-1, self.embed_dim)
        return hidden_states

class PatchMerger(nn.Module):
    def __init__(self, dim, context_dim, spatial_merge_size):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim)
        )

    def forward(self, x):
        x = self.ln_q(x).view(-1, self.hidden_size)
        x = self.mlp(x)
        return x

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000.0):
        super().__init__()

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)

        return freqs
    

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

class MLP(nn.Module):
    def __init__(self, dim, immediate_dim, bias=False, act_func='silu'):
        super().__init__()
        self.up_proj = nn.Linear(dim, immediate_dim, bias=bias)
        self.down_proj = nn.Linear(immediate_dim, dim, bias=bias)
        self.gate_proj = nn.Linear(dim, immediate_dim, bias=bias)

    def forward(self, x):
        x, gate = self.up_proj(x), self.gate_proj(x)
        x = F.silu(gate) * x
        x = self.down_proj(x)
        return x


def rotate_half(x):
    '''
     q1  q2  q3 q4 q5 q6
    --->
    -q4 -q5 -q5 q1 q2 q3
    '''
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2: ]
    return torch.cat((-x2, x1), -1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    这个咋跟 1d 的完全一样啊？？
    '''
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class VisionAttention(nn.Module):
    def __init__(self, dim, num_heads = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        '''
        hidden_states: [seq_length, hidden_size]
        没有 batch 的概念，都 packing 起来了？
        '''
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        # 后面都用 position_embeddings 代替 rotary_pos_embed 了

        if position_embeddings is None:
            raise ValueError
        else:
            cos, sin = position_embeddings

        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.full(
            [1, seq_length, seq_length],
            torch.finfo(q.dtype).min,         # torch.finfo 返回浮点数类型信息，这里是取了 q.dtype 类型的最小值，其实就是 -float('inf')
            device=q.device, dtype=q.dtype
        )

        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i-1]: cu_seqlens[i], cu_seqlens[i-1]: cu_seqlens[i]] = 0
            # NaViT Attention mask 区分不同图片 和 windown attention mask 区分不同滑窗
            # 但是这样用 mask 实现 window attention 实际上并没有减少计算量啊 https://github.com/QwenLM/Qwen2.5-VL/issues/1049

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(0, 1)
        output = output.reshape(seq_length, -1)
        output = self.proj(output)

        return output
    
class VisionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, immediate_dim):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, immediate_dim, bias=True)

    def forward(
        self,
        hidden_states,
        cu_seqlens,  # 用于计算 attention mask，for NaViT 和 window attention
        rotary_pos_emb=None,
        position_embeddings=None
    ):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )

        hidden_states = hidden_states + self.mlp(
            self.norm2(hidden_states)
        )

        return hidden_states


class VisionTransformer(nn.Module):
    def __init__(
        self,
        spatial_merge_size: int,
        temporal_patch_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        num_heads: int,
        out_hidden_size: int,      # LLM hidden_size
        intermediate_size: int,
        depth: int,
        fullatt_block_indexes: list[int],
        window_size: int,
        **kwargs  # 兼容其他额外参数
    ):
        super().__init__()

        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.window_size = window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size


        self.patch_embed = PatchEmbed(patch_size, temporal_patch_size, in_channels, hidden_size)

        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)   # 一半用于表示 h，一半用于表示 w

        self.blocks = nn.ModuleList([
            VisionBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(depth)
        ])

        self.merger = PatchMerger(
            dim=out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size
        )

    
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens



    def forward(self, hidden_states, grid_thw):
        """
        这个没有 batch 的概念，做了类似 navit 的 sequence packing？

        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
                其实应该是叫做 pixels_values (seq_len, hidden_size) (n_image*grid_t*grid_w*grid_h, in_channels*temporal_patch_size*spatial_patch_size**2) (seq_len, 1176)
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """

        hidden_states = self.patch_embed(hidden_states)     # (seq_len, 1176)  ---[3D Conv]--> (seq_len, hidden_size)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=hidden_states.device, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)


        seq_len, _ = hidden_states.size()
        # 变换一下 为了做 window attention ？
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        # RoPE 做同样的变换，保证 token 位置信息正确
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # 拼两个 head_dim // 2 的，一个用于表示 h， 一个用于表示 w
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0)
        cu_seqlens = cu_seqlens.to(grid_thw.dtype)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            # attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens_now)

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

        

if __name__ == '__main__':

    model_path = '/mnt/data/user/tc_ai/klara/models/open_mllm/qwen25_vl_3b/train-model/'
    config: Qwen2_5_VLConfig = Qwen2_5_VLConfig.from_pretrained(model_path)
    # print(config.vision_config)

    device = 'cuda'
    vit = VisionTransformer(**config.vision_config.to_dict()).to(device)
    
    inputs = torch.Tensor(1536, 1176).to(device)

    grid_thw = torch.tensor([[1, 32, 48]])      # [1, 98, 146]  太大了，80G 都爆显存

    outputs = vit(inputs, grid_thw)
    print(outputs.shape)
