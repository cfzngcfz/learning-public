# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 03:19:33 2024

@author: CC-i7-11700
"""
import torch, random
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt

# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print(features.size())
print(labels.size())
# plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)

# 读取数据集
batch_size = 10
data_iter = data.DataLoader(data.TensorDataset(features, labels), batch_size=batch_size, shuffle=True)
X, y = next(iter(data_iter))
print("X =\n", X)
print("y =\n", y)

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))
print(net)

# 参数初始化
"""
1. 通过`net[0]`选择网络中的第一个图层
2. 使用`weight.data`和`bias.data`方法访问参数
3. 使用替换方法`normal_`和`fill_`来重写参数值
"""
print(net[0].weight.data)
print(net[0].bias.data)
print(net[0].weight.data.normal_(0, 0.01))
print(net[0].bias.data.fill_(1))

# 验证模型和初始化参数
y_hat = torch.matmul(X, net[0].weight.data.T) + net[0].bias.data
print(torch.equal(y_hat, net(X)))

# 损失函数
loss = nn.MSELoss()

# 验证损失函数值
torch.sum((y_hat - y) ** 2)/y.numel()
loss(net(X) ,y)

# 优化算法
"""待优化的参数可通过net.parameters()获得"""
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    if isinstance(net, torch.nn.Module):
        net.train()         # 将模型设置为训练模式
    for X, y in data_iter:
        l = loss(net(X) ,y) # 计算损失函数loss（前向传播）
        trainer.zero_grad()
        l.backward()        # 进行反向传播来计算梯度
        trainer.step()      # 用优化器来更新模型参数
    
    if isinstance(net, torch.nn.Module):
        net.eval()          # 将模型设置为评估模式
    with torch.no_grad():
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
