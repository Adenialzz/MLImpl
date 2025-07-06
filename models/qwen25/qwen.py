import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2Config
from gqa import Block, RMSNorm
from rope import _compute_rope_params

class Qwen2_5(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()

        self.block_size = config.max_position_embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Block(
                hidden_dim=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                immediate_dim=config.intermediate_size
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.embed_tokens.weight = self.lm_head.weight      # 这应该在加载权重之后吧，而且是不是反了？我看 qwen 开源权重里只有前者

        cos_cached, sin_cached = _compute_rope_params(base=config.rope_theta, rope_dim=config.hidden_size//config.num_attention_heads, max_position_embeddings=config.max_position_embeddings)

        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)        # (B, T, D)

        cos = self.cos_cached[: T, :].unsqueeze(dim=0).expand(B, -1, -1)
        sin = self.sin_cached[: T, :].unsqueeze(dim=0).expand(B, -1, -1)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, None


    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_token_id=None, temperature=1.0, top_k=50):
        idx = input_ids

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size: ]    # 截取最近的 max_pos_ids(block_size) 个 token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature     # last hidden state

            if top_k is not None:
                v = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)

            if eos_token_id is not None:
                if idx_next == eos_token_id:
                    print('end due to eos token')
                    break

        return idx




