# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:46:36 2021

@author: CC-i7-8750H
"""
import torch
from torch import nn
from torch.nn import init
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #忽略libiomp5md.dll报错
#-----------------------------------------------------------------------------#
# 数据下载/读取
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    # download=True,
    download=False,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    # download=True,
    download=False,
    transform=ToTensor()
)
#-----------------------------------------------------------------------------#
# 读取批量数据
from torch.utils.data import DataLoader
batch_size = 100
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

num_inputs = training_data[0][0].numel()
num_outputs = 10
num_hiddens = 256
num_epochs = 5
learning_rate = 0.5
#-----------------------------------------------------------------------------#
# 1.定义模型
# 1.A
def mlp(X):
    Input = X.view((-1, num_inputs))
    Hidden_input = torch.matmul(Input, Weight1) + Bias1
    Hidden_output = torch.max(input=Hidden_input, other=torch.tensor(0.0))
    Ouput = torch.matmul(Hidden_output, Weight2) + Bias2
    return Ouput
# 1.B
MLP = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                    nn.ReLU(),
                    nn.Linear(num_hiddens, num_outputs),)

# 2. 参数初始化
# 2.A
Weight1 = torch.normal(mean=0, std=0.01, size=(num_inputs, num_hiddens), dtype=torch.float, requires_grad=True)
Bias1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
Weight2 = torch.normal(mean=0, std=0.01, size=(num_hiddens, num_outputs), dtype=torch.float, requires_grad=True)
Bias2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
# 2.B
for params in MLP.parameters():
    init.normal_(params, mean=0, std=0.01)
    
# 3.定义损失函数——交叉熵均值
loss_function = nn.CrossEntropyLoss()

# 4.定义参数优化算法——梯度下降算法
# 4.1
def sgd(params, learning_rate, batch_size):
    for param in params:
        param.data -= learning_rate * param.grad / batch_size
# 4.2
optimizer = torch.optim.SGD(MLP.parameters(), lr=learning_rate)

# 5.开始训练
procedure = 'A' # 'A':从零实现 or 'B':简洁实现

for epoch in range(num_epochs):
    # 5.1.训练
    train_correct = 0.0
    for X, y in train_dataloader:
        if procedure == 'A':
            # 5.1.1.模型计算
            pred = mlp(X)
            # 5.1.2.计算损失函数
            Loss = loss_function(pred, y)
            # 5.1.3.梯度清零
            optimizer.zero_grad()
            for params in [Weight1, Bias1, Weight2, Bias2]:
                if params.grad is not None:
                    params.grad.data.zero_()
            # 5.1.4.反向传播
            Loss.backward()
            # 5.1.5.参数优化-梯度下降算法
            sgd([Weight1,Bias1,Weight2,Bias2], learning_rate, batch_size)
        elif procedure == 'B':
            # 5.1.1.模型计算
            pred = MLP(X.view(X.shape[0], -1))
            # 5.1.2.计算损失函数
            Loss = loss_function(pred, y)
            # 5.1.3.梯度清零
            optimizer.zero_grad()
            for params in MLP.parameters():
                if params.grad is not None:
                    params.grad.data.zero_()
            # 5.1.4.反向传播
            Loss.backward()
            # 5.1.5.参数优化-梯度下降算法
            optimizer.step()
            
        train_correct += (pred.argmax(dim=1) == y).sum().item()
        
    # 5.2.测试
    test_correct = 0.0
    with torch.no_grad():
        for X, y in test_dataloader:
            if procedure == 'A':
                pred = mlp(X)
            elif procedure == 'B':
                pred = MLP(X.view(X.shape[0], -1))
                
            test_correct += (pred.argmax(dim=1) == y).sum().item()
    #结果展示
    print('epoch:', epoch+1,
          '| train accuracy:', train_correct/len(training_data),
          '| test accuracy:', test_correct/len(test_data))
#-----------------------------------------------------------------------------#
# 绘制激活函数 —— ReLU, sigmoid & tanh
import matplotlib.pyplot as plt
figure = plt.figure(figsize=(18, 12))

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

# ReLU函数
ax = figure.add_subplot(3, 2, 1)
y = x.relu()
ax.plot(x.detach(), y.detach())
ax.set_title('relu')

ax = figure.add_subplot(3, 2, 2)
y.sum().backward()
ax.plot(x.detach(), x.grad.detach())
ax.set_title('derivative of relu')

# sigmoid函数
ax = figure.add_subplot(3, 2, 3)
y = x.sigmoid()
ax.plot(x.detach(), y.detach())
ax.set_title('sigmoid')

ax = figure.add_subplot(3, 2, 4)
x.grad.zero_()
y.sum().backward()
ax.plot(x.detach(), x.grad.detach())
ax.set_title('derivative of sigmoid')

# tanh函数
ax = figure.add_subplot(3, 2, 5)
y = x.tanh()
ax.plot(x.detach(), y.detach())
ax.set_title('tanh')

ax = figure.add_subplot(3, 2, 6)
x.grad.zero_()
y.sum().backward()
ax.plot(x.detach(), x.grad.detach())
ax.set_title('derivative of tanh')