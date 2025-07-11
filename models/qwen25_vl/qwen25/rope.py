import torch
from torch import Tensor, nn

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


################################# 1D RoPE ##########################################

class RotaryPosistionEmbedding(nn.Module):
    def __init__(self, base, rope_dim, max_position_embeddings):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2) / rope_dim))

        self.register_buffer('inv_freq', inv_freq)
        self.inv_freq: torch.Tensor

    def forward(self, position_ids):
        ''''
        其实 1D RoPE 传入 seqlen 就行？这里和 3D RoPE 统一下接口

        position_ids torch.arange(seqlen)
        '''

        position_ids_expanded = position_ids  # (1, seqlen)  外面再 expand bsz
        inv_freq_expanded = self.inv_freq[:, None]  #  (dim//2, 1)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)

        emb = torch.cat((freqs, freqs), dim=-1)       # (T, dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + rotate_half(q) * sin
    k_embed = (k * cos) + rotate_half(k) * sin

    return q_embed, k_embed

################################# Multimodal 3D RoPE ##########################################

class MultimodalRotaryPositionEmbedding(nn.Module):
    def __init__(self, base, rope_dim, max_position_embeddings):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2) / rope_dim))
        self.register_buffer('inv_freq', inv_freq)
        self.inv_freq: Tensor

    def forward(self, position_ids):
        '''

        '''
        # MRoPE  与其他模型不同，Qwen2.5 VL 使用了 3D （时间、高度、宽度）多模态位置编码
        # 所以我们需要把 inv_freq 扩展成 (3, ...) 的形状
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()     # (3, bs, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        emb =  torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin



def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    '''
    Qwen2/2.5 VL 中的多模套旋转 3D RoPE

    多模态 3D 旋转位置嵌入是 1D 旋转位置嵌入的扩展。
    输入嵌入序列包含视觉（图像/视频）嵌入和文本嵌入，或者仅包含文本嵌入。
    对于视觉嵌入部分，我们分别在时间、高度和宽度维度上应用旋转位置嵌入。这里，我们将通道维度拆分为 3 个块，分别用于时间、高度和宽度旋转位置嵌入。
    对于文本嵌入部分，我们仅应用 1D 旋转位置嵌入。文本嵌入的三个旋转位置索引（时间、高度和宽度）始终相同，因此文本嵌入的旋转位置嵌入与现代 LLM 没有区别。

    mrope_section  定义了在整个 dim 中，分别各自有多少维度用于 时间、高度、宽度 的位置编码
    '''
    mrope_section = mrope_section * 2

    cos = torch.cat(
        [
            m[i % 3]
            for i, m in enumerate(cos.split(mrope_section, dim=-1))
        ],
        dim=-1
    ).unsqueeze(unsqueeze_dim)

    sin = torch.cat(
        [
            m[i % 3]
            for i, m in enumerate(sin.split(mrope_section, dim=-1))],
        dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed

