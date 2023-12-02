# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:28:27 2021

@author: CC-i7-8750H
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:49:07 2021

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
# 生成数据

# y = 1.2*x − 3.4*x**2 + 5.6*x**3 + 5
num_train, num_test = 10000, 1000
true_weight = [1.2, -3.4, 5.6]
true_bias = 5.
train_features = torch.randn((num_train, 1))
train_labels = true_weight[0]*train_features + true_weight[1]*torch.pow(train_features, 2) + true_weight[2]*torch.pow(train_features, 3) + true_bias + torch.normal(mean=0, std=0.01, size=(num_train, 1), dtype=torch.float)
test_features = torch.randn((num_test, 1))
test_labels = true_weight[0]*test_features + true_weight[1]*torch.pow(test_features, 2) + true_weight[2]*torch.pow(test_features, 3) + true_bias + torch.normal(mean=0, std=0.01, size=(num_test, 1), dtype=torch.float)
#-----------------------------------------------------------------------------#
# 读取批量数据
from torch.utils.data import DataLoader, TensorDataset
batch_size = 100
training_data = TensorDataset(train_features, train_labels)
test_data = TensorDataset(test_features, test_labels)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

num_inputs = training_data[0][0].numel()
num_outputs = 1
num_epochs = 40
learning_rate = 0.01
#-----------------------------------------------------------------------------#
# 1.定义模型
# model = nn.Linear(num_inputs, num_outputs)    #线性回归
class Polynomial(nn.Module):                    #多项式回归
    def __init__(self):
        super(Polynomial, self).__init__()
        self.params = nn.ParameterDict({
            'weight0': nn.Parameter(torch.randn(1)),
            'weight1': nn.Parameter(torch.randn(1)),
            'weight2': nn.Parameter(torch.randn(1)),
            'bias': nn.Parameter(torch.randn(1))})
    def forward(self, x):
        x = self.params.weight0*x + self.params.weight1*torch.pow(x, 2) + self.params.weight2*torch.pow(x, 3) + self.params.bias
        return x
model = Polynomial()
# 3.定义损失函数——交叉熵均值
loss_function = nn.MSELoss()
# 4.定义参数优化算法——梯度下降算法
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 5.开始训练
train_loss, test_loss = [], []
for epoch in range(num_epochs):
    # 5.1.训练
    for X, y in train_dataloader:
        # 5.1.1.模型计算
        pred = model(X)
        # 5.1.2.计算损失函数
        Loss = loss_function(pred, y)
        # 5.1.3.梯度清零
        optimizer.zero_grad()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        # 5.1.4.反向传播
        Loss.backward()
        # 5.1.5.参数优化-梯度下降算法
        optimizer.step()

    train_loss.append(loss_function(model(train_features), train_labels).item()) #记录epoch训练后的训练集的损失函数值
    test_loss.append(loss_function(model(test_features), test_labels).item())    #记录epoch训练后的测试集的损失函数值

import matplotlib.pyplot as plt
figure = plt.figure(figsize=(8, 6))
ax = figure.add_subplot(1,1,1)
ax.semilogy([ii for ii in range(num_epochs)], train_loss, linestyle='-', label='train loss')
ax.semilogy([ii for ii in range(num_epochs)], test_loss, linestyle=':', label='test loss')
ax.set_xlabel('features')
ax.set_ylabel('labels')
ax.legend(loc='upper right')
#-----------------------------------------------------------------------------#
"""
线性回归作为对照：研究过拟合和欠拟合
小结:
0) 训练误差(training error): 模型在训练数据集上表现出的误差
    泛化误差(generalization error): 模型在任意一个测试数据样本上表现出的误差的期望，并常常通过测试数据集上的误差来近似
1) 由于无法从训练误差估计泛化误差，一味地降低训练误差并不意味着泛化误差一定会降低。
   机器学习模型应关注降低泛化误差。
2) 可以使用验证数据集来进行模型选择,不可以使用测试数据选择模型
3) 欠拟合指模型无法得到较低的训练误差，过拟合指模型的训练误差远小于它在测试数据集上的误差。
4) 应选择复杂度合适的模型并避免使用过少的训练样本。
"""
# K折交叉验证的程序还未写过