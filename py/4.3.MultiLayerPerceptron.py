# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:32:51 2024

@author: CC-i7-11700
"""

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

# 1. 数据下载 & 读取
# source: 3.5.3  整合所有组件
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    num_workers = 0
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, 
                            shuffle=True,             # 随机打乱所有样本
                            num_workers=num_workers), # 使用 num_workers 个进程来读取数据
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=num_workers))

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
# --------------------------------------------------------------------------- #
# 2. 定义模型 & 参数初始化

# # source: 4.2.1. 初始化模型参数
# num_inputs, num_outputs, num_hiddens = 784, 10, 256
# W1 = nn.Parameter(torch.randn(
#     num_inputs, num_hiddens, requires_grad=True) * 0.01)
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# W2 = nn.Parameter(torch.randn(
#     num_hiddens, num_outputs, requires_grad=True) * 0.01)
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
# params = [W1, b1, W2, b2]

# # source: 4.2.2. 激活函数
# def relu(X):
#     a = torch.zeros_like(X)
#     return torch.max(X, a)

# # source: 4.2.3. 模型
# def net(X):
#     X = X.reshape((-1, num_inputs))
#     H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
#     return (H@W2 + b2)

# source: 4.3.1. 模型
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
# --------------------------------------------------------------------------- #
# 3. 定义损失函数

# source: 4.2.4. 损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# --------------------------------------------------------------------------- #
# 4. 优化算法

lr = 0.1

# # source: 4.2.5  训练
# trainer = torch.optim.SGD(params, lr=lr)

# source: 4.3.1  模型
trainer = torch.optim.SGD(net.parameters(), lr=lr)
# --------------------------------------------------------------------------- #
# 5. 训练 & 评估

# source: 3.6.5 分类精度
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y): # 评估
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()                          # 将模型设置为评估模式
    metric = Accumulator(2)                 # 累加器: 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), # 当前分类正确次数
                       y.numel())           # 当前样本数量
    return metric[0] / metric[1]            # 返回测试集的分类准确率

# source: 3.6.6 训练
def train_epoch_ch3(net, train_iter, loss, trainer):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()                         # 将模型设置为训练模式
    metric = Accumulator(3)                 # 累加器: 训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)                  # 计算损失函数loss(前向传播)
        if isinstance(trainer, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            trainer.zero_grad()             # 梯度清零
            l.mean().backward()             # 反向传播计算梯度
            trainer.step()                  # 用优化器更新模型参数
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            trainer(X.shape[0])
        metric.add(float(l.sum()),          # 训练损失总和
                   accuracy(y_hat, y),      # 训练准确度总和
                   y.numel())               # 样本数
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2] # 返回训练集的平均损失和分类准确率

def train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer):
    """训练模型"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, trainer) # 训练模型
        test_acc = evaluate_accuracy(net, test_iter)                    # 评估模型
        print(train_metrics)
        print(test_acc)
    train_loss, train_acc = train_metrics
    
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# source: 4.2.5  训练
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)