# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:13:22 2021

@author: CC-i7-8750H
"""
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
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
    # """
    # transforms=ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片
    # 或者数据类型为np.uint8的NumPy数组
    # 转换为尺寸为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor
    # 如果不进行转换则返回Image图片
    # """
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    # """
    # label -> one-hot
    # tensor.scatter_(dim=0, index=torch.tensor(y), value=1) #将tensor的第dim维 第index个元素 的值修改为value
    # """
)

training_data2 = datasets.FashionMNIST(
    root="data",
    train=True,
    # download=True,
    download=False,
    # transform=ToTensor(),
    # """
    # transforms=ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片
    # 或者数据类型为np.uint8的NumPy数组
    # 转换为尺寸为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor
    # 如果不进行转换则返回Image图片
    # """
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    # """
    # label -> one-hot
    # tensor.scatter_(dim=0, index=torch.tensor(y), value=1) #将tensor的第dim维 第index个元素 的值修改为value
    # """
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    # download=True,
    download=False,
    transform=ToTensor()
)
#-----------------------------------------------------------------------------#
# 基本信息
print("type of data:",type(training_data),
      '\n  number of samples:',len(training_data))
print('type of the first sample:',type(training_data[0][0]),
      '\n  size of this sample:',training_data[0][0].size(),
      '\n  dtype of this sample:',training_data[0][0].dtype,
      '\n  label of a sample:',training_data[0][1])
print('type of the first sample:',type(training_data2[0][0]),
      '\n  label of this sample:',training_data2[0][1])
#-----------------------------------------------------------------------------#
#随机显示9个sample
labels_map = {0: "T-Shirt",1: "Trouser",2: "Pullover",3: "Dress",4: "Coat",
              5: "Sandal",6: "Shirt",7: "Sneaker",8: "Bag",9: "Ankle Boot",}
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")  #tensor1.squeeze()将tensor1中所有为1的维度去除
    # <=> plt.imshow(img.view((28, 28)))
    # <=> plt.imshow(img.squeeze().numpy()) #imshow的对象既可以是tensor，也可以是numpy
    # <=> plt.imshow(img.view((28, 28)).numpy()) 
plt.show()
#-----------------------------------------------------------------------------#
# 读取批量数据
from torch.utils.data import DataLoader, TensorDataset

# 数据
inputs = torch.randn(100, 2, dtype=torch.float32)
targets = torch.squeeze(torch.mm(inputs,torch.tensor([2, -3.4]).view(2,1)) +4.2) + torch.normal(mean=0., std=0.01, size=(inputs.size(0),), dtype=torch.float)
batch_size = 10 
# 组合输入和目标值，构造DataLoader
inputs_and_targets = TensorDataset(inputs, targets) 
data_dataloader = DataLoader(inputs_and_targets, batch_size=batch_size, shuffle=True)
"""
<=>
def DataLoader2(batch_size, inputs, targets):
    indices = torch.randperm(len(inputs))
    for ii in range(0, len(inputs), batch_size):
        batch_indices = indices[ii: min(ii + batch_size, len(inputs))]
        yield  inputs.index_select(dim=0, index=batch_indices), targets.index_select(dim=0, index=batch_indices)
        # yield 生成一个迭代器
"""

# DataLoader: 随机打乱顺序后生成批量数据
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=0)
"""
batch_size: 批量大小
shuffle: 是否打乱顺序
num_workers: 读取数据进程数量
"""
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)
# train_dataloader 是一个随机打乱顺序后的可迭代的对象

# 查看DataLoaders 生成的可迭代对象
for data in train_dataloader:  #train_dataloader是可迭代对象，则可用for循环遍历
    print(data[0].size())
    print(data[1].size())

for img,label in train_dataloader:
    print(img.size())
    print(label.size())
    
for img,label in iter(train_dataloader): #同上等效
    print(img.size())
    print(label.size())

for img,label in train_dataloader:
    print(img.size())
    print(label.size())

# 查找前4批数据的第一个数据在原书数据中的序号，并绘图
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
index = 0
fig = plt.figure()
for img,label in train_dataloader:
    # print(img.size())
    # print(label.size())
    for ii in range(60000):
        if torch.equal(img[0][0],training_data[ii][0][0]): # 判断两个tensor是否相同
            break
    print('batch '+str(ii//64)+' number '+str(ii%64))
    ax0 = fig.add_subplot(2,2,index+1)
    ax0.imshow(img[0][0], cmap="gray")
    ax0.set_title('batch '+str(ii//64)+' number '+str(ii%64)+' label '+str(label[0].item()))
    index += 1
    if index == 4:
        break

# 打乱顺序后的第一个数据，查询其在原数据中的序号，并绘图显示
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
# 使用iter()进行访问train_dataloader，返回一个迭代器，然后使用next()访问
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img2 = train_features[0].squeeze() #将tensor中所有为1的维度删掉
label2 = train_labels[0]
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.imshow(img2, cmap="gray")
ax0.set_title(labels_map[label2.item()])
for ii in range(60000):
    if torch.equal(img2,training_data[ii][0][0]):  # 判断两个tensor是否相同
        break
print('next iter 的结果与 '+str(ii//64)+'批第'+str(ii%64)+'个相同')




