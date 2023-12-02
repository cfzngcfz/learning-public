# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:22:16 2021

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
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()

    return mask * X / keep_prob
# X = torch.arange(16).view(2, 8)
# print(dropout(X, 0))
# print(dropout(X, 0.5))
# print(dropout(X, 1))
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
drop_prob = 0.2        # inverted dropout probability
#-----------------------------------------------------------------------------#
# 1.定义模型
# 1.A
def mlp(X, is_training):
    Input = X.view((-1, num_inputs))
    Hidden_input = torch.matmul(Input, Weight1) + Bias1
    Hidden_output = torch.max(input=Hidden_input, other=torch.tensor(0.0))
    if is_training:                                         # 只在训练模型时使用丢弃法
        Hidden_output = dropout(Hidden_output, drop_prob)   # 在隐含层后添加丢弃层
    Ouput = torch.matmul(Hidden_output, Weight2) + Bias2
    return Ouput
# 1.B
MLP = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                    nn.ReLU(),
                    nn.Dropout(drop_prob),                  # 丢弃层
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
procedure = 'B' # 'A':从零实现 or 'B':简洁实现

for epoch in range(num_epochs):
    # 5.1.训练
    train_correct = 0.0
    if procedure == 'A':
        if 'is_training' in mlp.__code__.co_varnames: # 如果有is_training这个参数
            is_training = True
    elif procedure == 'B':
        MLP.train() #训练模式,开启dropout
    for X, y in train_dataloader:
        if procedure == 'A':
            # 5.1.1.模型计算
            pred = mlp(X, is_training)
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
    if procedure == 'A':
        if 'is_training' in mlp.__code__.co_varnames: # 如果有is_training这个参数
            is_training = False
    elif procedure == 'B':
        MLP.eval() # 评估模式,关闭dropout
    with torch.no_grad():
        for X, y in test_dataloader:
            if procedure == 'A':
                pred = mlp(X, is_training)
            elif procedure == 'B':
                pred = MLP(X.view(X.shape[0], -1))
                
            test_correct += (pred.argmax(dim=1) == y).sum().item()
    #结果展示
    print('epoch:', epoch+1,
          '| train accuracy:', train_correct/len(training_data),
          '| test accuracy:', test_correct/len(test_data))
#-----------------------------------------------------------------------------#
"""
多层感知机例子：研究过拟合的倒置丢弃法（inverted dropout）
小结:
1) 倒置丢弃法(克服过拟合方法之一)：在训练时，以drop_prob概率，将指定层的部分输出变为0，其他输出除以1−p做拉伸，保持输出的期望值不变，
2) 丢弃法只在训练模型时使用
"""