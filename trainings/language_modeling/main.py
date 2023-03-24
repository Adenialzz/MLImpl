import torch
import argparse
from contextlib import nullcontext

import sys; sys.path.append('../../models/gpt/')
from gpt import GPT, GPTConfig
from lm_engine import train



def get_cfg():
    parser = argparse.ArgumentParser()

    # training config
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grad_accumulation_steps', type=int, default=5)
    parser.add_argument('--dtype', type=str, default='float16', choices=('float32', 'bfloat16', 'float16'))
    
    # gpt model config
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embed', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)

    # misc
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_progress_bar', '-npb', action='store_false')

    cfg = parser.parse_args()
    return cfg

def main(cfg):
    torch.manual_seed(1337)
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    model_args = dict(
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_embed=cfg.n_embed,
            block_size=cfg.block_size, bias=cfg.bias, vocab_size=None,
            dropout=cfg.dropout
    )
    model_args['vocab_size'] = 50304
    gpt_config = GPTConfig(**model_args)
    model = GPT(gpt_config)

    ptdtype = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
            }[cfg.dtype]
    ctx =  nullcontext() if device_type=='cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type=='cuda' and cfg.dtype=='float16'))
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), cfg.device)
    train(model, ctx, scaler, optimizer, cfg)


if __name__ == '__main__':
    cfg = get_cfg()
    main(cfg)
