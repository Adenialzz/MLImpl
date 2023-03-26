import torch
import os.path as osp
import time
from utils import get_batch, get_lr


def train(model, ctx, scaler, optimizer, ddp, master_process, cfg, resume_info=None):
    X, Y = get_batch('train', cfg)
    local_iter_num = 0
    raw_model = model.module if ddp else model
    if resume_info is not None:
        iter_num = resume_info[0]
        best_val_loss = resume_info[1]
    else:
        iter_num = 0
        best_val_loss = 1e9
    t0 = 0
    running_mfu = -1.0
    while True:
        lr = cfg.lr if cfg.no_lr_decay else get_lr(iter_num, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % cfg.eval_interval == 0 and master_process:
            losses = eval_loss(model, ctx, cfg)
            print(f"step {iter_num}: train_loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
            if losses['val'] < best_val_loss or cfg.always_save_ckpt:
                best_val_loss = losses['val']
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': raw_model.config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'train_config': cfg,
                }
                if iter_num > 0:
                    print(f'saving checkpoint to {cfg.out_dir}')
                    torch.save(ckpt, osp.join(cfg.out_dir, 'ckpt.pt'))

        for micro_step in range(cfg.grad_accumulation_steps):
            if ddp:
                # 在多卡并行且使用多次前向积累梯度时，需要在最后一次前向时同步梯度
                # 官方的做法是使用model.no_sync()上下文管理器
                # 通过查看改上下文管理器中具体做的事情，可以用下面一行代替
                model.require_backward_grad_sync = (micro_step == cfg.grad_accumulation_steps - 1)
                pass
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train', cfg)  # 模型在GPU上进行计算时，CPU立即开始异步读取下一批次的数据
            scaler.scale(loss).backward()
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0 and master_process:
            lossf = loss.item()  # 作为CPU-GPU同步点
            if local_iter_num >= 5:  # 让训练循环稍微稳定下来
                mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.grad_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg.max_iters:
            break


@torch.no_grad()
def eval_loss(model, ctx, cfg):
    results = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split, cfg)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        results[split] = losses.mean()
    model.train()
    return results




