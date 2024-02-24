# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 03:20:15 2024

@author: CC-i7-11700
"""

import torch, torchvision, time
from torch import nn
from torchvision import transforms
from torch.utils import data
from matplotlib import pyplot as plt
from IPython import display

# 下载数据集
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../../temp", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST( root="../../temp", train=False, transform=trans, download=True)
print("训练集中的样本数量 =", len(mnist_train), ", 测试集中的样本数量", len(mnist_test))
print("size of input of first sample =", mnist_train[0][0].shape)
print("label of first sample =", mnist_train[0][1])

# 读取数据集
## 部分数据可视化
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
batch_size = 10
data_iter = data.DataLoader(mnist_train,
                            batch_size=batch_size,
                            shuffle=True,  # 随机打乱所有样本
                            num_workers=4) # 使用4个进程来读取数据                                                              

X, y = next(iter(data_iter))
show_images(X.reshape(batch_size, 28, 28), 2, int(batch_size/2), titles=get_fashion_mnist_labels(y))

## 多进程来读取数据对比
start = time.time()
for X, y in data.DataLoader(mnist_train, batch_size=256):
    continue
end = time.time()
print(end - start, "sec")

start = time.time()
for X, y in data.DataLoader(mnist_train, batch_size=256, num_workers=4):
    continue
end = time.time()
print(end - start, "sec")

## 训练和测试数据
batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=4)

# 定义模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
print(net)

# 参数初始化
for ii in range(len(net)):
    print(ii, type(net[ii]) == nn.Linear)
net[1].weight.data.normal_(0.0, 0.01) # 方法1
# nn.init.normal_(net[1].weight, mean=0.0, std=0.01) # 方法2
print(net[1].weight.data)

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 验证
## 验证 softmax 函数
X2 = torch.normal(0, 1, (2, 5))
print("X2 =\n", X2)
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
X_prob = softmax(X2)
print("X_prob =\n", X_prob)
### 分步验证
X_exp = torch.exp(X2)
print("X_exp =\n", X_exp)
partition = X_exp.sum(1, keepdim=True)
print("partition =\n", partition)
X_prob = X_exp / partition
print("X_prob =\n", X_prob)
print("sum of X_prob = ", X_prob.sum(1))

## 验证模型和初始参数
X, y = next(iter(data_iter))
Output = torch.matmul(X.reshape((-1, net[1].weight.data.T.shape[0])), net[1].weight.data.T) + net[1].bias.data
print(torch.equal(Output, net(X)))

## 验证loss
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
y_hat = softmax(Output)
print("y =", y)
print(cross_entropy(y_hat, y))
print(loss(net(X) ,y))

# 优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
## 准备(累加器和动画)
class Accumulator: # 累加器
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()
        
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

## 定义分类正确函数
"""分类概率->分类结果"""
def accuracy(y_hat, y):
    """计算分类正确次数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
### 分步验证
if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    y_hat = y_hat.argmax(axis=1)
print("y_hat =", y_hat) 
print("y =    ", y) # y_hat & y defined in 验证loss
cmp = y_hat.type(y.dtype) == y # 由于等式运算符“==”对数据类型很敏感， 因此我们将y_hat的数据类型转换为与y的数据类型一致
print("cmp =  ", cmp)
print("分类正确的次数 =", float(cmp.type(y.dtype).sum()))

## 使用训练集，训练模型
def train(net, train_iter, loss, optimizer):
    if isinstance(net, torch.nn.Module):
        net.train()                    # 将模型设置为训练模式
    metric = Accumulator(3)            # 累加器
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)             # 计算损失函数loss（前向传播）
        optimizer.zero_grad()
        l.mean().backward()            # 进行反向传播来计算梯度
        optimizer.step()               # 用优化器来更新模型参数

        metric.add(float(l.sum()),     # 训练损失总和
                   accuracy(y_hat, y), # 训练准确度总和
                   y.size(0))          # 样本数
    return metric[0] / metric[2], metric[1] / metric[2] # 返回训练集的平均损失和分类准确率

## 使用测试集，评估模型
def evaluate(net, test_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()                          # 将模型设置为评估模式
    metric = Accumulator(2)                 # 累加器
    with torch.no_grad():
        for X, y in test_iter:
            metric.add(accuracy(net(X), y), # 当前分类正确次数
                       y.size(0))           # 当前样本数量

    return metric[0] / metric[1]           # 返回测试集的分类准确率

## 运行多个迭代周期。 在每个迭代周期结束时，利用test_iter访问到的测试数据集对模型进行评估
num_epochs = 10
animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=["train loss", "train acc", "test acc"])

for epoch in range(num_epochs):
    train_metrics = train(net, train_iter, loss, optimizer) # 训练模型
    test_acc = evaluate(net, test_iter)                     # 评估模型
    print(epoch + 1, "train loss =", train_metrics[0], ", train acc =", train_metrics[1], ", test acc =", test_acc)
    animator.add(epoch + 1, train_metrics + (test_acc,))

train_loss, train_acc = train_metrics
assert train_loss < 0.5, train_loss
assert train_acc <= 1 and train_acc > 0.7, train_acc
assert test_acc <= 1 and test_acc > 0.7, test_acc

# 预测
num_samples = 8
X, y = next(iter(data.DataLoader(mnist_test, num_samples, shuffle=True)))
labels = get_fashion_mnist_labels(y)
outputs = get_fashion_mnist_labels(net(X).argmax(axis=1))
titles = [true +'\n' + pred for true, pred in zip(labels, outputs)]
show_images(torch.squeeze(X), 1, num_samples, titles=titles)