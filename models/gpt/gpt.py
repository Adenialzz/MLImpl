import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    # 如果PyTorch>=2.0，可以使用官方实现的F.scaled_dot_product_attention
    # 基于CUDA核实现，速度更快
    # 此处手动实现
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout

        # causal mask用于保证仅计算当前位置左侧的注意力
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # scaled
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # mask
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 计算完成后，重装所有头的输出
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024   # 最大序列长度 max_len
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
            ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # 将输入映射矩阵`transformer.wte.weight`和输出映射矩阵`lm_head.weight`的参数绑定
        # ref：# https://paperswithcode.com/method/weight-tying

        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print('number of parameters: %.2fM' % (self.get_num_params() / 1e6, ))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 如果给定了目标，则可以同时计算损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段的小优化：仅对最后一个位置使用lm_head进行前向推理
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[: block_size])
        for blk in self.transformer.h:
            blk.attn.bias = blk.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        override_args = override_args or {}
        # 只有dropout可以被重写
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print(f'loading weights from pretraiend gpt: {model_type}')

        config_args = {
                'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M Params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M Params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M Params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M Params
        }[model_type]

        print('focing vocab_size=50257, block_size=1024, bias=True')
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # 丢弃 mask/buffer，这不是参数

        # 获取huggingface/transformers的权重参数
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制参数，保证所有参数的名称与形状是对应的
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 丢弃buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 丢弃mask

        # openai权重使用Conv1D，而我们的实现使用Linear，因此在导入以下权重时需要转置
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys)
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, \
                print(f'transposed model params {k} shape misaligned: hf: {sd_hf[k].shape[::-1]}, model: {sd[k].shape}')
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, \
                print(f'model params {k} shape misaligned: hf: {sd_hf[k].shape}, model: {sd[k].shape}')
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        该函数用于配置GPT的优化器，之所以这么冗长是因为需要根据是否需要权重衰减将GPT的参数分为两部分
        并配置不同的优化器
        """

        # 将所有参数根据是否需要参数衰减分为两部分，白名单需要，黑名单不需要
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # 参数全名
                # 由于`named_modules`和`named_parameters`是递归的，
                # 因此我们会多次看到同一个张量p，
                # 这样做我们可以知道张量p来自哪个父模块
                if pn.endswith('bias'):
                    # 所有bias都需要权重衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 白名单中的参数需要权重衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # 黑名单中的参数不需要权重衰减
                    no_decay.add(fpn)

        # 注意：`transformer.wte.weight`和`lm_head.weight`分别出现在no_day和decay集中。
        # 然而，在我们的实现中它们是绑定的，
        # 由于`named_parameters()`不会返回重复项，因此它只会返回第一个出现项，
        # 因此，让我们手动从衰减集decay中删除`lm_head.weight`。
        decay.remove('lm_head.weight')

        # 验证每个参数属于且只属于decay/no_decay中的一个
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # 创建PyTorch优化器
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # fused选项可以使AdamW优化器的速度更快
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        '''
        输入一个由索引组成条件序列(即形状为(b, t)的序列)，并进行max_new_times次预测，每次将预测的token添加到条件序列之后，继续进行预测。使用generate之前，需要将模型设置为推理模式model.eval()。
        '''
        for _ in range(max_new_tokens):
            # 如果序列长度超过最大长度block_size，截取后半部分长度为block_size的序列
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size]
            # 模型根据现有序列进行推理
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)

            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样出的索引添加到序列中，并继续
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

