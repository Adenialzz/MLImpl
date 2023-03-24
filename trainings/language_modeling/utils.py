import torch
import numpy as np
import os.path as osp

def get_batch(split, cfg):
    data = np.memmap(osp.join(cfg.data_dir, f'{split}.bin'), dtype='uint16', mode='r')
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size, ))
    x = torch.stack([torch.from_numpy((data[i: i+cfg.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1: i+1+cfg.block_size]).astype(np.int64)) for i in ix])
    if 'cuda' in cfg.device:
        x = x.pin_memory().to(cfg.device, non_blocking=True)
        y = y.pin_memory().to(cfg.device, non_blocking=True)
    else:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
    return x, y


