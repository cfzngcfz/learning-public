# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:18:24 2021

@author: CC-i7-8750H
"""

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
num_train, num_test, num_inputs = 1000, 100, 100
true_weight = torch.ones(num_inputs, 1)*0.01
true_bias = 0.05
train_features = torch.randn((num_train, num_inputs))
train_labels = torch.matmul(train_features, true_weight) + true_bias + torch.normal(mean=0, std=0.01, size=(num_train, 1), dtype=torch.float)
test_features = torch.randn((num_test, num_inputs))
test_labels = torch.matmul(test_features, true_weight) + true_bias + torch.normal(mean=0, std=0.01, size=(num_test, 1), dtype=torch.float)
#-----------------------------------------------------------------------------#
# 读取批量数据
from torch.utils.data import DataLoader, TensorDataset
batch_size = 100
training_data = TensorDataset(train_features, train_labels)
test_data = TensorDataset(test_features, test_labels)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

num_outputs = 1
num_epochs = 100
learning_rate = 0.003
WeightDecay = 0             # 权重衰减 0 vs 3 vs 5
#-----------------------------------------------------------------------------#
# 1.定义模型
model = nn.Linear(num_inputs, num_outputs)    #线性回归
# 2.参数初始化
init.normal_(model.weight, mean=0, std=1)
init.normal_(model.bias, mean=0, std=1)
# 3.定义损失函数——交叉熵均值
loss_function = nn.MSELoss()
# 4.定义参数优化算法——梯度下降算法
optimizer = torch.optim.SGD([{'params': model.weight, 'weight_decay': WeightDecay}, # 如果未指定参数，则使用最外层的默认参数
                             {'params': model.bias}],
                            lr=learning_rate, weight_decay=0)
# <=>
# optimizer_weight = torch.optim.SGD(params=[model.weight], lr=learning_rate, weight_decay=WeightDecay) # 对权重参数衰减
# optimizer_bias = torch.optim.SGD(params=[model.bias], lr=learning_rate)  # 不对偏差参数衰减

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
        # <=>
        # optimizer_weight.zero_grad()
        # optimizer_bias.zero_grad()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        # 5.1.4.反向传播
        Loss.backward()
        # 5.1.5.参数优化-梯度下降算法
        optimizer.step()
        # <=>
        # optimizer_weight.step()
        # optimizer_bias.step()
        
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
多元线性回归例子：研究过拟合的权重衰减
小结:
1) 权重衰减(克服过拟合方法之一)通过给损失函数添加"L2范数正则化"惩罚,使训练的模型参数值较接近0
2) 可以通过优化器中的weight_decay超参数来指定
"""
