import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import argparse
from engine import train, validate

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view([-1, 1 * 28 * 28])
        x = self.fc1(x)
        x = self.act(x)
        out = self.fc2(x)
        return out

def get_cfg():
    parser = argparse.ArgumentParser()

    # training config
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--writer_dir', type=str, default='log/')
    parser.add_argument('--out_dir', type=str, default='out/')
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    model = MnistModel().to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ])

    train_set = torchvision.datasets.MNIST(root=cfg.data_root, train=True, transform=pipeline, download=True)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_set = torchvision.datasets.MNIST(root=cfg.data_root, train=False, transform=pipeline, download=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    writer = SummaryWriter(cfg.writer_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)
    for e in range(cfg.epochs):
        train_loss = train(e, model, optimizer, train_loader, cfg.device)
        val_loss, val_acc = validate(e, model, test_loader, cfg.device)
        writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, e)
        writer.add_scalar('val acc', val_acc, e)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_cfg': cfg
        }
        torch.save(checkpoint, os.path.join(cfg.out_dir, f'ckpt_e{e}.pth'))

    

if __name__ == '__main__':
    cfg = get_cfg()
    main(cfg)

