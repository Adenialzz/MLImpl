import torch
import torch.nn as nn

# pytorch docs: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
# layer norm paper: https://arxiv.org/abs/1607.06450

# features shape: (batch_size, max_len, hidden_dim)
class LayerNorm(nn.Module):
    def __init__(self, num_dims, eps=1e-5):
        super().__init__()
        shape = (num_dims, )
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x):
        # 与bn主要的区别就是在归一化所在的维度不同
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.gamma * x_hat + self.beta
        return y
