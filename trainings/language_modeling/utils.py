import torch
import numpy as np
import math
import os.path as osp
from gpt import GPT, GPTConfig

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

def load_vocab(meta_path):
    if not osp.exists(meta_path):
        return None
    else:
        import pickle
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f'found vocab_size = {meta_vocab_size} at {meta_path}')
    return meta_vocab_size

def load_gpt_model(cfg):
    model_args = dict(
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_embed=cfg.n_embed,
            block_size=cfg.block_size, bias=cfg.bias, vocab_size=None,
            dropout=cfg.dropout
    )
    if cfg.init_from == 'scratch':
        print('Initializing A New GPT Model.')
        if cfg.meta_vocab_size is None:
            print('Defaulting Vocab Size of GPT2 is 50304 (50257 rounded up for efficiency)')
            model_args['vocab_size'] = 50304
        else:
            model_args['vocab_size'] = cfg.meta_vocab_size
        gpt_config = GPTConfig(**model_args)
        model = GPT(gpt_config)
    elif cfg.init_from == 'resume':
        print(f"resuming gpt from {cfg.resume_dir}")
        ckpt_path = osp.join(cfg.resume_dir, 'ckpt.pt')
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        ckpt_model_args = ckpt['model_args']
        for key in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
            model_args[key] = ckpt_model_args[key]
        gpt_config = GPTConfig(**model_args)
        model = GPT(gpt_config)
        state_dict = ckpt['model']
        unwaneted_prefix = '_orig_mod'
        for k, _ in list(state_dict.items()):
            if k.startswith(unwaneted_prefix):
                state_dict[k[len(unwaneted_prefix): ]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = ckpt['iter_num']
        best_val_loss = ckpt['best_val_loss']
    elif cfg.init_from.startswith('gpt2'):
        print(f"Initing from OpenAI GPT2 weights: {cfg.init_from}")
        override_args = dict(dropout=cfg.dropout)
        model = GPT.from_pretrained(cfg.init_from, override_args)
        for key in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
            model_args[key] = getattr(model.config, key)

    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args['block_size'] = cfg.block_size

    return model

def get_lr(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.lr * it / cfg.warmup_iters
    elif it > cfg.lr_decay_iters:
        return cfg.min_lr
    else:
        decay_ratio = (it - cfg.warmup_iters) / (lr_decay_iters - warmup_iters)
        assert decay_ratio >= 0 and decay_ratio <= 1
        coeff = 0.5 * (1.0 - math.cos(math.pi * cfg.decay_ratio))
        return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


