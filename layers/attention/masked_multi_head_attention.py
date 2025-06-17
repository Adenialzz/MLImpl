import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ref: https://blog.csdn.net/weixin_54338498/article/details/136421557
class MaskedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        hidden_dim: int,
        n_heads: int
    ):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads

        self.w_q = nn.Linear(q_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(kv_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(kv_dim, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, q_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        bsz, q_len, _ = hidden_states.shape
        _, kv_len, _ = encoder_hidden_states.shape

        q = self.w_q(hidden_states)
        k = self.w_k(encoder_hidden_states)
        v = self.w_v(encoder_hidden_states)

        # prepare attn_mask to attn_bias: 0/False->-inf
        attention_bias = torch.zeros(q_len, kv_len, dtype=q.dtype)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attention_bias.masked_fill_(attention_mask.logical_not()), -float('inf')  # mask 0 to -inf
            else:
                attention_bias += attention_mask

        # split heads：(batch_size, seq_len, hidden_dim)->(batch_size, heads, seq_len, head_dim)
        q = q.view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # q @ k.T
        attention_score = torch.matmul(q, k.transpose(2, 3))

        # add bias
        # print(attention_score.shape, attention_bias.shape)
        attention_score = attention_score + attention_bias  # 这俩维度不一样，这样可以加。但是不能 += ？  广播机制

        # / sqrt(d) and softmax
        attention_score = F.softmax(attention_score / math.sqrt(self.head_dim), dim=-1)

        # get output
        output = torch.matmul(attention_score, v)

        output = output.view(bsz, -1, self.n_heads * self.head_dim).contiguous()

        output = self.w_o(output)

        return output



if __name__ == '__main__':
    attention = MaskedMultiHeadAttention(q_dim=1024, kv_dim=1024, hidden_dim=768, n_heads=12)
    inputs = torch.Tensor(16, 512, 1024) # bsz, seq_len, hidden_dim

    outputs = attention(inputs)




