# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 01:04:09 2024

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

# # source: 3.6.1  初始化模型参数
# num_inputs = 784
# num_outputs = 10
# W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# b = torch.zeros(num_outputs, requires_grad=True)

# # source: 3.6.2  定义softmax操作
# def softmax(X):
#     X_exp = torch.exp(X)
#     partition = X_exp.sum(1, keepdim=True)
#     return X_exp / partition  # 这里应用了广播机制

# def net(X):
#     return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# source: 3.7.1  初始化模型参数
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
# --------------------------------------------------------------------------- #
# 3. 定义损失函数

# # source: 3.6.4  定义损失函数
# def loss(y_hat, y): # 优化的损失函数
#     # cross_entropy
#     return - torch.log(y_hat[range(len(y_hat)), y])

# source: 3.7.2  重新审视Softmax的实现
loss = nn.CrossEntropyLoss(reduction='none')
# --------------------------------------------------------------------------- #
# 4. 优化算法

# # source: 3.6.6  训练
# lr = 0.1

# def sgd(params, lr, batch_size):
#     """Minibatch stochastic gradient descent."""
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()
            
# def trainer(batch_size):
#     return sgd([W, b], lr, batch_size)

# source: 3.7.3  优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# ---------------------------------------------------------------------------
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
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()                         # 将模型设置为训练模式
    metric = Accumulator(3)                 # 累加器: 训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)                  # 计算损失函数loss(前向传播)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()             # 梯度清零
            l.mean().backward()             # 反向传播计算梯度
            updater.step()                  # 用优化器更新模型参数
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),          # 训练损失总和
                   accuracy(y_hat, y),      # 训练准确度总和
                   y.numel())               # 样本数
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2] # 返回训练集的平均损失和分类准确率

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) # 训练模型
        test_acc = evaluate_accuracy(net, test_iter)                    # 评估模型
        print(train_metrics)
        print(test_acc)
    train_loss, train_acc = train_metrics
    
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
# ---------------------------------------------------------------------------
num_epochs = 10
# source: 3.7.4 训练
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)