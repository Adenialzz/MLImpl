import torch
import torch.nn as nn
import numpy as np

from layer_norm import LayerNorm

bs, max_len, hidden_dim = 4, 16, 256

x = torch.randn(bs, max_len, hidden_dim)

ln_pytorch = nn.LayerNorm(hidden_dim, elementwise_affine=True)   # `elementwise_affine`参数可以选择是否有可学习的参数weight和bias
y_pytorch = ln_pytorch(x)

ln_customized = LayerNorm(hidden_dim)
y_customized = ln_customized(x)

np.testing.assert_allclose(y_pytorch.detach().numpy(), y_customized.detach().numpy(), rtol=1e-6)
