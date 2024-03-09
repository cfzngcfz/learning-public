# -*- coding: utf-8 -*-

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
dropout1, dropout2 = 0.2, 0.5

# # source: 4.6.4 从零开始实现
# def dropout_layer(X, dropout):
#     assert 0 <= dropout <= 1
    
#     # 在本情况中，所有元素都被丢弃
#     if dropout == 1:
#         return torch.zeros_like(X)
    
#     # 在本情况中，所有元素都被保留
#     if dropout == 0:
#         return X
    
#     mask = (torch.rand(X.shape) > dropout).float()
    
#     return mask * X / (1.0 - dropout) # 公式 4.6.1

# # source: 4.6.4.2 定义模型

# class Net(nn.Module):
#     def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
#                  is_training = True):
#         super(Net, self).__init__()
#         self.num_inputs = num_inputs
#         self.training = is_training
#         self.lin1 = nn.Linear(num_inputs, num_hiddens1)
#         self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
#         self.lin3 = nn.Linear(num_hiddens2, num_outputs)
#         self.relu = nn.ReLU()

#     def forward(self, X):
#         H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
#         # 只有在训练模型时才使用dropout
#         if self.training == True:
#             # 在第一个全连接层之后添加一个dropout层
#             H1 = dropout_layer(H1, dropout1)
#         H2 = self.relu(self.lin2(H1))
#         if self.training == True:
#             # 在第二个全连接层之后添加一个dropout层
#             H2 = dropout_layer(H2, dropout2)
#         out = self.lin3(H2)
#         return out

# num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
# net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# source: 4.6.5  简洁实现
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
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

# source: 4.3.1  模型
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# # todo: 权重衰减
# wd = 0.2
# trainer = torch.optim.SGD([
#     {"params":net[0].weight,'weight_decay': wd},
#     {"params":net[0].bias}], lr=lr)
# trainer = torch.optim.SGD([
#     {'params': net.weight, 'weight_decay': wd}, # 如果未指定参数，则使用最外层的默认参数
#     {'params': net.bias}],
#     lr=lr, weight_decay=0)
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