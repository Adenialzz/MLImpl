import torch
import torch.nn as nn

import numpy as np

def train(epoch, model, optimizer, train_loader, device, master_process=True):
    loss_func = nn.CrossEntropyLoss()
    loss_list = []
    model.train()
    for idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_func(logits, targets)
        loss.backward()
        loss_list.append(loss.item())
        optimizer.step()
        if idx % 50 == 0 and master_process:
            print(f'Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.3f}')
    mean_loss = np.mean(loss_list)
    return mean_loss

def validate(epoch, model, val_loader, device):
    loss_list, acc_list = [], []
    loss_func = nn.CrossEntropyLoss()
    model.eval()
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
    mean_loss = np.mean(loss_list)
    mean_acc = np.mean(acc_list)
    print(f"TEST     Epoch: {epoch}, Acc: {mean_acc * 100:.2f}%, Loss: {mean_loss:.3f}")
    return mean_loss, mean_acc

