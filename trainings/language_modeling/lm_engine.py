import torch
from utils import get_batch
def train(model, ctx, scaler, optimizer, cfg):
    X, Y = get_batch('train', cfg)

    while True:
        for micro_step in range(cfg.grad_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                print(loss.item())
            scaler.scale(loss).backward()
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

