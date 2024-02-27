# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 03:19:33 2024

@author: CC-i7-11700
"""
import torch
from torch import nn
from torch.utils import data

# 1. 数据生成 & 读取
# source: 3.3.1  生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# source: 3.3.2  读取数据集
batch_size = 10
data_iter = data.DataLoader(data.TensorDataset(features, labels),
                            batch_size=batch_size,
                            shuffle=True) # 是否希望数据迭代器对象在每个迭代周期内打乱数据
# X, y = next(iter(data_iter))
# --------------------------------------------------------------------------- #
# 2. 定义模型 & 参数初始化

# # source: 3.2.3  初始化模型参数
# w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# # source: 3.2.4  定义模型
# def linreg(X, w, b):
#     """线性回归模型"""
#     return torch.matmul(X, w) + b

# def net(X):
#     return linreg(X, w, b)

# source: 3.3.3  定义模型
net = nn.Sequential(nn.Linear(2, 1))

# source: 3.3.4  初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# --------------------------------------------------------------------------- #
# 3.损失函数

# # source: 3.2.5  定义损失函数
# def loss(y_hat, y):
#     """均方损失"""
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# source: 3.3.5  定义损失函数
loss = nn.MSELoss()
# --------------------------------------------------------------------------- #
# 4.优化算法

# # source: 3.2.6  定义优化算法
# def sgd(params, lr, batch_size):
#     """小批量随机梯度下降"""
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()

# def trainer(batch_size):
#     return sgd([w, b], 0.03, batch_size)

# source: 3.3.6  定义优化算法
"""待优化的参数可通过net.parameters()获得"""
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# --------------------------------------------------------------------------- #
# 5. 训练 & 评估

num_epochs = 3
for epoch in range(num_epochs):
    if isinstance(net, torch.nn.Module):
        net.train()         # 将模型设置为训练模式
    
    for X, y in data_iter:
        l = loss(net(X) ,y) # 计算损失函数loss(前向传播)
        if isinstance(trainer, torch.optim.Optimizer):
            trainer.zero_grad()
            l.backward()        # 反向传播计算梯度
            trainer.step()      # 使用优化器更新模型参数
        else:
            l.sum().backward()
            trainer(X.shape[0])
    
    if isinstance(net, torch.nn.Module):
        net.eval()          # 将模型设置为评估模式
    
    with torch.no_grad():
        if isinstance(loss, torch.nn.MSELoss):
            l = loss(net(features), labels) # 监控训练过程
        else:
            l = loss(net(features), labels).mean().item()
       
        print(f'epoch {epoch + 1}, loss {l:f}')

if isinstance(net, torch.nn.Module):
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
else:
    w = w.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = b.data
    print('b的估计误差：', true_b - b)
