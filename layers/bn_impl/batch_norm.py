import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims, eps=1e-5, momentum=0.9):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # gamma和beta是可学习的参数，分别表示缩放修正量和平移修正量
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # running_mean和running_var不是通过梯度更新，而是通过滑动平均的方式维护整个数据集的均值和方差，在测试阶段直接使用
        self.running_mean = torch.zeros(shape)
        self.running_var = torch.ones(shape)
        self.eps = eps
        self.momentum = momentum

    def batch_norm(self, X):
        # 判断当前是训练阶段还是测试阶段
        if not torch.is_grad_enabled():   # 测试阶段，直接使用训练阶段在整个训练集上计算得到的均值与方差进行归一化
            X_hat = (X - self.running_mean) / torch.sqrt(self.running_var + self.eps)  
        else:  # 训练阶段，先计算当前批次的均值和方差，进行归一化，并通过滑动平均的方式更新整个训练集的均值和方差
            assert len(X.shape) in (2, 4) # 仅支持num_dims=2 || 4, 分别对应fc层和conv层的的BN操作
            # 首先计算当前批次的均值和方差
            if len(X.shape) == 2:  # for fc
                # 对于fc，计算特征维上的均值和方差
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else: # for conv
                # 对于conv，计算通道维上的均值和方差
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            # 进行归一化
            X_hat = (X - mean) / torch.sqrt(var + self.eps)
            # 滑动平均更新整个训练集的均值和方差
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        Y = self.gamma * X_hat + self.beta  # 缩放和平移修正, gamma和beta是可学习的修正量参数
        return Y

    def forward(self, X):
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_var = self.running_var.to(X.device)
        Y = self.batch_norm(X)
        return Y
