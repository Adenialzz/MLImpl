import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
import os

from tensorboardX import SummaryWriter
from ddp_utils import init_ddp, stop_ddp
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
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        device, master_process, seed_offest = init_ddp()
        # logging, save_ckpt only save on master_process
    else:
        device = cfg.device
        master_process = True
        seed_offest = 0

    torch.manual_seed(1337 + seed_offest)
    model = MnistModel().to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    raw_model = model.module if ddp else model

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ])

    train_set = torchvision.datasets.MNIST(root=cfg.data_root, train=True, transform=pipeline, download=True)
    train_sampler = DistributedSampler(train_set) if ddp else None
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=not ddp, sampler=train_sampler)
    test_set = torchvision.datasets.MNIST(root=cfg.data_root, train=False, transform=pipeline, download=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    writer = SummaryWriter(cfg.writer_dir)
    if master_process:
        os.makedirs(cfg.out_dir, exist_ok=True)
    for e in range(cfg.epochs):
        train_loss = train(e, model, optimizer, train_loader, device, master_process)
        if master_process:  # only log and save checkpoint on master_process
            val_loss, val_acc = validate(e, model, test_loader, device)
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, e)
            writer.add_scalar('val acc', val_acc, e)
            checkpoint = {
                'state_dict': raw_model.state_dict(),  # remove key name prefix `.module`
                'optimizer': optimizer.state_dict(),
                'train_cfg': cfg
            }
            torch.save(checkpoint, os.path.join(cfg.out_dir, f'ckpt_e{e}.pth'))
    
    if ddp:
        stop_ddp()

if __name__ == '__main__':
    cfg = get_cfg()
    main(cfg)

