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
num_epochs = 5
learning_rate = 0.1
#-----------------------------------------------------------------------------#
# 1.定义模型
# 1.A
def net(X):
    X1= torch.mm(X.view((-1, num_inputs)), Weight) + Bias
    return X1
# 1.B
linear = nn.Linear(num_inputs, num_outputs)

# 2. 参数初始化
# 2.A
Weight = torch.normal(mean=0, std=0.01, size=(num_inputs, num_outputs), dtype=torch.float, requires_grad=True)
Bias = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
# 2.B
init.normal_(linear.weight, mean=0, std=0.01)
init.constant_(linear.bias, val=0) 

# 3.定义损失函数——交叉熵均值
# 3.A
def cross_entropy(X, y):
    # 交叉熵
    # y'_i = x_1*w_1i +x_2*w_2i + ... + x_n*w_ni + bi
    # y_hat_i = exp(y'_i)/(exp(y'_1)+exp(y'_2)+...+exp(y'_n))
    # 性质1： exp()单调递增, 不改变相对大小 => argmax(y') = argmax(y_hat)，即不改变预测类型输出 
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    y_hat = X_exp / partition
    # cross entropy = - y_1*log(y_hat_1) - y_2*log(y_hat_2) - ... - y_n*log(y_hat_n)
    # 性质2：if y_i = 1 and y_other = 0, then cross entropy 仅与 y_hat_i 的相关，与y_hat_other无关
    #        即交叉熵只关心对正确类别的预测概率
    cross_entropy = - torch.log(y_hat.gather(1, y.view(-1, 1)))
    return cross_entropy.mean()
# 3.B
loss_function = nn.CrossEntropyLoss()

# 4.定义参数优化算法——梯度下降算法
# 4.A
def sgd(params, learning_rate, batch_size):
    for param in params:
        param.data -= learning_rate * param.grad / batch_size
# 4.B
optimizer = torch.optim.SGD(linear.parameters(), lr=learning_rate)

# 5.开始训练
procedure = 'B' # 'A':从零实现 or 'B':简洁实现

for epoch in range(num_epochs):
    # 5.1.训练
    train_correct = 0.0
    for X, y in train_dataloader:
        if procedure == 'A':
            # 5.1.1.模型计算
            pred = net(X)
            # 5.1.2.计算损失函数
            Loss = cross_entropy(pred, y)
            # 5.1.3.梯度清零
            if Weight.grad is not None:
                Weight.grad.data.zero_()
            if Bias.grad is not None:
                Bias.grad.data.zero_()
            # 5.1.4.反向传播
            Loss.backward()
            # 5.1.5.参数优化-梯度下降算法
            sgd([Weight,Bias], learning_rate, batch_size)
        elif procedure == 'B':
            # 5.1.1.模型计算
            pred = linear(X.view(X.shape[0], -1))
            # 5.1.2.计算损失函数
            Loss = loss_function(pred, y)
            # 5.1.3.梯度清零
            optimizer.zero_grad()
            if linear.weight.grad is not None:
                linear.weight.grad.data.zero_()
            if linear.bias.grad is not None: 
                linear.bias.grad.data.zero_()
            # 5.1.4.反向传播
            Loss.backward()
            # 5.1.5.参数优化-梯度下降算法
            optimizer.step()
        train_correct += (pred.argmax(dim=1) == y).sum().item()
        
    # 5.2.验证
    test_correct = 0.0
    with torch.no_grad():
        for X, y in test_dataloader:
            if procedure == 'A':
                pred = net(X)
            elif procedure == 'B':
                pred = linear(X.view(X.shape[0], -1))
            test_correct += (pred.argmax(dim=1) == y).sum().item()
    #结果展示
    print('epoch:', epoch+1,
          '| train accuracy:', train_correct/len(training_data),
          '| test accuracy:', test_correct/len(test_data))
#-----------------------------------------------------------------------------#
# 6.预测
X, y = iter(test_dataloader).next()
if procedure == 'A':
    y_hat = net(X).argmax(dim=1)
elif procedure == 'B':
    y_hat = linear(X.view(X.shape[0], -1)).argmax(dim=1)
labels_map = {0: "T-Shirt",1: "Trouser",2: "Pullover",3: "Dress",4: "Coat",
              5: "Sandal",6: "Shirt",7: "Sneaker",8: "Bag",9: "Ankle Boot",}
import matplotlib.pyplot as plt
figure = plt.figure(figsize=(18, 6))
for ii in range(5):
    img = X[ii]
    true_label = y[ii].item()
    predict_label = y_hat[ii].item()
    ax = figure.add_subplot(1, 5, ii+1)
    ax.set_title('True:'+labels_map[true_label]+'\nPred:'+labels_map[predict_label])
    ax.imshow(img.squeeze())  
