# -*- coding: utf-8 -*-
"""
Created on Thu May 20 01:00:17 2021

@author: CC-i7-8750H
"""

import torch
from torch import nn
from torch.nn import init

# import random
# import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #忽略libiomp5md.dll报错
#-----------------------------------------------------------------------------#
# 读取数据
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques
import pandas
train_data = pandas.read_csv('./data/kaggle_house/train.csv')
test_data = pandas.read_csv('./data/kaggle_house/test.csv')

# 数据预处理
def DataPreprocessing(data):
    indices = data.dtypes[data.dtypes != 'object'].index #数值列的索引
    data[indices] = data[indices].apply(lambda x: (x - x.mean()) / (x.std()))
    data[indices] = data[indices].fillna(0)
    data = pandas.get_dummies(data, dummy_na=True)
    # pandas.get_dummies将非数值列转化为one hot encode
    # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
    return data

all_data = pandas.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
all_data = DataPreprocessing(all_data).values
features = torch.tensor(all_data[0:train_data.shape[0]], dtype=torch.float)
test_features = torch.tensor(all_data[train_data.shape[0]:], dtype=torch.float)
labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)
# labels = torch.log(labels)
#-----------------------------------------------------------------------------#
from torch.utils.data import DataLoader, TensorDataset
num_inputs = features.size(1)
num_examples = features.size(0)
# 超参数
num_k = 5
batch_size = 64
learning_rate = 5
num_epochs = 100
WeightDecay = 0
#-----------------------------------------------------------------------------#
dataset = TensorDataset(features, labels) # 将特征和标签组合
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# 1.定义模型
model = nn.Linear(num_inputs, 1)
# model = nn.Sequential(nn.Linear(num_inputs, 200),
#                       nn.ReLU(),
#                       nn.Linear(200, 100),
#                       nn.ReLU(),
#                       nn.Linear(100, 1),)
# 2. 参数初始化
for param in model.parameters():
    init.normal_(param, mean=0, std=0.01)
# 3.定义损失函数——MSE
loss_function = nn.MSELoss()
# 4.定义参数优化算法——梯度下降算法
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WeightDecay)
# 5.开始训练
train_loss = []
# test_loss = []
for epoch in range(num_epochs):
    # 5.1.训练
    for X, y in dataloader:
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
    train_loss.append(loss_function(model(features), labels).item()) #记录epoch训练后的训练集的损失函数值
    # test_loss.append(loss_function(model(test_features), test_labels).item())    #记录epoch训练后的测试集的损失函数值

preds = model(test_features).detach()

import matplotlib.pyplot as plt
figure = plt.figure(figsize=(8, 6))
ax = figure.add_subplot(1,1,1)
ax.semilogy([ii for ii in range(num_epochs)], train_loss, linestyle='-', label='train loss')
# ax.semilogy([ii for ii in range(num_epochs)], test_loss, linestyle=':', label='test loss')
ax.set_xlabel('features')
ax.set_ylabel('labels')
ax.legend(loc='upper right')
#-----------------------------------------------------------------------------#
# K-fold cross-validation
def get_k_fold_data(num_k, index_fold, features, labels):
    
    assert num_k > index_fold >= 0
    k_start = [0]
    train_features, train_labels = None, None
    for mm in range(num_k):
        if mm < features.size(0)%num_k:
            k_start.append(k_start[-1]+features.size(0)//num_k+1)
        else:
            k_start.append(k_start[-1]+features.size(0)//num_k)
        cur_features, cur_labels = features[k_start[-2]:k_start[-1]], labels[k_start[-2]:k_start[-1]]
        if mm == index_fold:
            valid_features, valid_labels = cur_features, cur_labels
        elif train_features is None:
            train_features, train_labels = cur_features, cur_labels
        else:
            train_features = torch.cat((train_features, cur_features), dim=0)
            train_labels = torch.cat((train_labels, cur_labels), dim=0)
    return train_features, train_labels, valid_features, valid_labels


train_loss_sum, valid_loss_sum = 0, 0
for index_fold in range(num_k):
    train_features, train_labels, valid_features, valid_labels = get_k_fold_data(num_k, index_fold, features, labels)
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 1.定义模型
    model = nn.Linear(num_inputs, 1)
    # 2. 参数初始化
    for param in model.parameters():
        init.normal_(param, mean=0, std=0.01)    
    # 3.定义损失函数——MSE
    loss_function = nn.MSELoss()        
    # 4.定义参数优化算法——梯度下降算法
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WeightDecay)
    # 5.开始训练
    train_loss, valid_loss = [], []
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
        # 5.2.验证
        train_loss.append(loss_function(model(train_features), train_labels).item()) #记录epoch训练后的训练集的损失函数值
        valid_loss.append(loss_function(model(valid_features), valid_labels).item())    #记录epoch训练后的测试集的损失函数值
    print('Fold',index_fold,
          ': train loss', train_loss[-1],
          'valid loss', valid_loss[-1])
    train_loss_sum += train_loss[-1]
    valid_loss_sum += valid_loss[-1]
print(num_k,'-fold cross-validation: avg train loss', train_loss_sum/num_k,
      'avg valid loss', valid_loss_sum/num_k)
