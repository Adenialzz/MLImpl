import torch
from torch.distributed import init_process_group, destroy_process_group
import os


def init_ddp():
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_progress = ddp_rank == 0
    seed_offest = ddp_rank
    return device, master_progress, seed_offest

def stop_ddp():
    destroy_process_group()
