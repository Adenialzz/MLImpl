import torch
import torch.nn as nn

import numpy as np

def train(epoch, model, optimizer, data_loaders, cfg):
    loss_func = nn.CrossEntropyLoss()
    train_loader, val_loader = data_loaders
    for idx, (images, targets)  in enumerate(train_loader):
        images = images.to(cfg.device)
        targets = targets.to(cfg.device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_func(logits, targets)
        loss.backward()
        optimizer.step()
        if idx % cfg.log_interval == 0:
            print(f'Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.3f}')
    test(epoch, model, val_loader, cfg.device)

def test(epoch, model, val_loader, device):
    loss_list, acc_list = [], []
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            cur_loss = loss_func(output, targets)
            loss_list.append(cur_loss.item())
            pred = output.max(dim = -1)[-1]
            cur_acc = pred.eq(targets).float().mean()
            acc_list.append(cur_acc.item())
    print(f"TEST     Epoch: {epoch}, Acc: {np.mean(acc_list):.2f}, Loss: {np.mean(loss_list):.3f}")

