

# Attention 层实现

Transformer 的核心机制：注意力层的实现

## 实现

- 经典的 transformers GPT2 Attention：[link](https://github.com/huggingface/transformers/blob/2507169bf658e39e6ffe89a04b32e3729b218b73/src/transformers/models/gpt2/modeling_gpt2.py#L155-L351)

- torch 2.x scaled_dot_product_attention 等价示例 [torch_sdpa.py](./torch_sdpa.py)

- 手写 Attention [masked_multi_head_attention.py](./masked_multi_head_attention.py)

## 注意

1. multi-head

2. attention mask / attention bias / causal mask ...

3. cross attention (q: hidden_states, k,v: encoder_hidden_states)
