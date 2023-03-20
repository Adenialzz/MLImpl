from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import torch
import numpy as np
 
#准备数据集
def get_dataloader(train, batch_size):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean = (0.1307,),std = (0.3081,))
        ])
 
    dataset = MNIST(root='./data', train=train, transform=transform_fn, download=True)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=train)
    return data_loader
 
def train(epoch, model, optimizer, batch_size, device):#epoch表示几轮
    data_loader = get_dataloader(True, batch_size)#获取数据加载器
    loss_func = nn.CrossEntropyLoss()
    for idx, (image, target)  in enumerate(data_loader):#idx表示data_loader中的第几个数据，元组是data_loader的数据
        image = image.to(device)
        target = target.to(device)
        optimizer.zero_grad()#将梯度置0
        output = model(image)#调用模型，得到预测值
        loss = loss_func(output, target) #调用损失函数，得到损失,是一个tensor
        loss.backward()#反向传播
        optimizer.step()#梯度的更新
        if idx % 50 == 0:
            print(f'epoch: {epoch}\t step: {idx}\t loss: {loss.item():.3f}')
 
def test(model, batch_size, device):
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False, batch_size=batch_size)#获取测试集
    loss_func = nn.CrossEntropyLoss()
    for image, target in test_dataloader:
        with torch.no_grad():#不计算梯度
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = loss_func(output, target)
            loss_list.append(loss.item())
            #计算准确率，output大小[batch_size,10] target[batch_size] batch_size是多少组数据，10列是每个数字概率
            pred = output.max(dim = -1)[-1]#获取最大值位置
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.item())
    print("平均准确率：", np.mean(acc_list), "平均损失：",np.mean(loss_list))
 
