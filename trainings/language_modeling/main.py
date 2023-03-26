import torch
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse
from contextlib import nullcontext

from lm_engine import train
from utils import load_vocab, load_gpt_model


def get_cfg():
    parser = argparse.ArgumentParser()
    # misc
    parser.add_argument('--data_dir', type=str, default='data/data/shakespeare')
    parser.add_argument('--meta_path', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='test_out/')
    parser.add_argument('--no_lr_decay', action='store_false')
    parser.add_argument('--init_from', type=str, default='scratch', choices=('scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--no_progress_bar', '-npb', action='store_false')

    # training config
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--warmup_iters', type=int, default=2000)
    parser.add_argument('--lr_decay_iters', type=int, default=600000)
    parser.add_argument('--max_iters', type=int, default=600000)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grad_accumulation_steps', type=int, default=5)
    parser.add_argument('--dtype', type=str, default='float16', choices=('float32', 'bfloat16', 'float16'))
    parser.add_argument('--eval_interval', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_iters', type=int, default=10)

    
    # gpt model config
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embed', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)

    cfg = parser.parse_args()
    return cfg

def main(cfg):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=cfg.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        cfg.device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(cfg.device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        cfg.grad_accumulation_steps *= 8
    torch.manual_seed(1337 + seed_offset)
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    if device_type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg.meta_vocab_size = load_vocab(cfg.meta_path)
    model = load_gpt_model(cfg).to(cfg.device)

    ptdtype = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
            }[cfg.dtype]
    ctx =  nullcontext() if device_type=='cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type=='cuda' and cfg.dtype=='float16'))
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), cfg.device)

    if master_process:
        os.makedirs(cfg.out_dir, exist_ok=True)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    train(model, ctx, scaler, optimizer, ddp, master_process, cfg)


if __name__ == '__main__':
    cfg = get_cfg()
    main(cfg)
