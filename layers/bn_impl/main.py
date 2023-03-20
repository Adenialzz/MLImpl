import torch
import torch.nn as nn
from engine import train, test
from batch_norm import BatchNorm


def get_bn(num_features, num_dims, method):
    assert method in ('pytorch', 'customized', 'none')
    if method == 'pytorch':
        if num_dims == 2:
            return nn.BatchNorm1d(num_features)
        elif num_dims == 4:
            return nn.BatchNorm2d(num_features)
    elif method == 'customized':
        return BatchNorm(num_features, num_dims)
    else:
        return nn.Identity()

if __name__ == "__main__":
    device = 'mps:0'
    bn_method = 'customized'
    lenet = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), get_bn(6, 4, bn_method), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), get_bn(16, 4, bn_method), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120), get_bn(120, 2, bn_method), nn.Sigmoid(),
            nn.Linear(120, 84), get_bn(84, 2, bn_method), nn.Sigmoid(),
            nn.Linear(84, 10)
        ).to(device)
    optimizer = torch.optim.Adam(lenet.parameters(), lr=0.1)
    batch_size = 256
    for e in range(10):
        train(e, lenet, optimizer, batch_size, device)
    test(lenet, batch_size, device)


