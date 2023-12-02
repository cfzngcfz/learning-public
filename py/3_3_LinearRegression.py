import torch
from torch import nn
from torch.nn import init

# import random
# import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #忽略libiomp5md.dll报错
#-----------------------------------------------------------------------------#
# 生成训练数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = torch.squeeze(torch.mm(features,torch.tensor(true_w).view(2,1)) +true_b) + torch.normal(mean=0., std=0.01, size=(num_examples,), dtype=torch.float)

# 数据展示
import matplotlib.pyplot as plt
figure = plt.figure(figsize=(18, 6))
ax0 = figure.add_subplot(121)
ax0.scatter(features[:,0].numpy(), labels.numpy(), marker='o')
ax1 = figure.add_subplot(122)
ax1.scatter(features[:,1].numpy(), labels.numpy(), marker='o')
#-----------------------------------------------------------------------------#
# 读取批量数据
from torch.utils.data import DataLoader, TensorDataset
batch_size = 10
dataset = TensorDataset(features, labels) # 将特征和标签组合
data_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

learning_rate = 0.03
num_epochs = 5
#-----------------------------------------------------------------------------#
# 1.定义模型
# 1.A
def linreg(X, Weight, Bias):
    return torch.mm(X, Weight) + Bias
# 1.B
linear = nn.Linear(num_inputs, 1)

# 2. 参数初始化
# 2.A
Weight = torch.normal(mean=0, std=0.01, size=(num_inputs, 1), dtype=torch.float32, requires_grad=True)
Bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)
# 2.B
init.normal_(linear.weight, mean=0, std=0.01)
init.constant_(linear.bias, val=0) # <=> net[0].bias.data.fill_(0)

# 3.定义损失函数——MSE
# 3.A
def loss_mse(y_hat, y):
    loss = (y_hat - y.view(y_hat.size())) ** 2 
    return loss.mean()
# 3.B
loss_function = nn.MSELoss()


# 4.定义参数优化算法——梯度下降算法
# 4.A
def sgd(params, learning_rate, batch_size):
    for param in params:
        param.data -= learning_rate * param.grad / batch_size # 注意这里更改param时用的param.data
# 4.B
optimizer = torch.optim.SGD(linear.parameters(), lr=learning_rate)
"""
# 1.不同子网络设置为不同的学习率
optimizer = torch.optim.SGD(
    [{'params': net.subnet1.parameters()}, # 如果未指定学习率，则使用最外层的默认学习率，即lr=0.03
     {'params': net.subnet2.parameters(), 'lr': 0.01}],
    lr=0.03)
# 2.调整学习率(待实例了解具体用法)
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
"""

# 5.开始训练
procedure = 'B' # 'A':从零实现 or 'B':简洁实现

for epoch in range(1, num_epochs + 1):
    for X, y in data_dataloader:
        if procedure == 'A':
            # 5.1.模型计算
            output = linreg(X, Weight, Bias) 
            # 5.2.计算损失函数
            Loss = loss_mse(output, y)
            # 5.3.梯度清零
            if Weight.grad is not None:
                Weight.grad.data.zero_()
            if Bias.grad is not None:
                Bias.grad.data.zero_()
            # 5.4.反向传播
            Loss.backward()
            # 5.5.参数优化
            sgd([Weight,Bias], learning_rate, batch_size)
        elif procedure == 'B':
            # 5.1.模型计算
            output = linear(X)
            # 5.2.计算损失函数
            Loss = loss_function(output, y.view(-1, 1))
            # 5.3.梯度清零
            optimizer.zero_grad()
            if linear.weight.grad is not None:
                linear.weight.grad.data.zero_()
            if linear.bias.grad is not None: 
                linear.bias.grad.data.zero_()
            # 5.4.反向传播
            Loss.backward()
            # 5.5.参数优化
            optimizer.step()
            
    print('epoch %d, loss: %f' % (epoch, Loss.item()))

if procedure == 'A':
    print('\n Real weight:',true_w,
          '\n Learning weight:', Weight.data,
          '\n Real bias:', true_b,
          '\n Learning bias:', Bias.data )
elif procedure == 'B':
    print('\n Real weight:',true_w,
          '\n Learning weight:', linear.weight.data,
          '\n Real bias:', true_b,
          '\n Learning bias:', linear.bias.data)
