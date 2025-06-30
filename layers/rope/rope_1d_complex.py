import torch
from torch import Tensor, nn

def compute_freq_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:

    # freqs: size=dim//2,   10000^{2i/d}
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # end: max_seq_len
    t = torch.arange(end, device=freqs.device)
    
    # (end,) X (dim/2,) -> (end, dim/2)
    freqs = torch.outer(t, freqs)

    # torch.polar
    #     是 PyTorch 中的一个函数，用于根据给定的绝对值（模）和角度（相位）创建复数张量。
    #     它将极坐标形式的复数转换为直角坐标形式的复数张量。
    # 函数签名
    #     torch.polar(abs, angle) → Tensor
    # 参数
    #     abs (Tensor): 输入张量，表示复数的绝对值（模）。必须是非负的。
    #     angle (Tensor): 输入张量，表示复数的角度（相位），以弧度为单位。

    # 通过复平面极坐标计算sin和cos
    # 通过polar函数,将长度为1的向量逆时针旋转freqs度
    # freqs_cis的shape是(end, dim/2)
    # 每个元素是一个复数,实部是cos,虚部是sin
    freq_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freq_cis



def apply_rope(q: Tensor, k: Tensor, freq_cis: Tensor):
    '''
    q: [bs, seq_len, num_heads, head_dim]
    k: [bs, seq_len, num_heads, head_dim]
    freq_cis: [seq_len, head_dim//2]
    '''

    # xq.shape = [bsz, seqlen, self.n_local_heads, self.head_dim]
    # 先把 head_dim 维拆成两份，分别作为复数的实部和虚部
    # xq_.shape = [bsz, seqlen, self.n_local_heads, self.head_dim//2 , 2]
    # torch.view_as_complex用于将二维向量转换为复数域 torch.view_as_complex即([q0,q1]) -> (q0 + q1 j)
    # 所以经过view_as_complex变换后xq_.shape = [bsz, seqlen, self.n_local_heads, self.head_dim//2]
    print("q input", q.shape)
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[: -1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[: -1], -1, 2))
    print("q complex", q_complex.shape)

    print("freq_cis before expand", freq_cis.shape)
    freq_cis_expand = freq_cis.unsqueeze(0).unsqueeze(2)    # expand bs and num_head, for broadcast
    print("freq_cis after expand", freq_cis.shape)

    # 旋转
    q_out = torch.view_as_real(q_complex * freq_cis_expand).flatten(-2)
    k_out = torch.view_as_real(k_complex * freq_cis_expand).flatten(-2)

    print("q output", q_out.shape)
    return q_out.to(q.dtype), k_out.to(k.dtype)


class ROPE1D(nn.Module):
    def __init__(self, dim: int, max_position: int):
        super().__init__()
        self.freq_cis = compute_freq_cis(dim, max_position)
    
    def forward(self, q, k):
        freq_cis = self.freq_cis[: q.size(1)]

        return apply_rope(q, k, freq_cis)


if __name__ == '__main__':
    bs, seq_len, n_head, head_dim = 32, 1024, 12, 128
    q = Tensor(bs, seq_len, n_head, head_dim)
    k = Tensor(bs, seq_len, n_head, head_dim)
    max_position = 10240

    # freq_cis = compute_freq_cis(head_dim, max_position)
    # q, k = apply_rope(q, k, freq_cis[: seq_len])

    rope_1d = ROPE1D(head_dim, max_position)

    q, k = rope_1d(q, k)