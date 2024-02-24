# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 01:43:08 2021

@author: CC-i7-8750H
"""
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
from torch import nn
import torch.nn.init as init # 引入初始化模块
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #忽略libiomp5md.dll报错

# 数据下载
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    # download=True,
    download=False,
    transform=ToTensor(), # img -> 无量纲化+float格式
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    # # label -> one-hot
    # # tensor.scatter_(dim=0, index=torch.tensor(y), value=1) #将tensor的第dim维 第index个元素 的值修改为value
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    # download=True,
    download=False,
    transform=ToTensor()
)


# list,numpy,tensor相互转化
img_tensor, label = training_data[0]
print(type(img_tensor))
img_numpy = img_tensor.numpy()                  # tensor -> numpy
print(type(img_numpy))
print(img_numpy.shape)
img_list = img_numpy.tolist()                   # numpy -> list
print(type(img_list))

print(type(  numpy.array(img_list)  ))          # list -> numpy
print(type(  img_numpy.tolist()  ))             # numpy -> list
print(type(  torch.Tensor(img_list)  ))         # list -> tensor
print(type(  torch.from_numpy(img_numpy)  ))    # numpy -> tensor

# 数据可视化
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")  #将tensor中所有为1的维度删掉
plt.show()

# # 自定义数据
# import pandas as pd
# from torchvision.io import read_image

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         sample = {"image": image, "label": label}
#         return sample


# DataLoaders 随机打乱顺序后生成批量数据
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
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


# 检验 GPU or CPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using {} device'.format(device))

#=============================================================================#
# 建模
# 输入
input_image = torch.rand(3,28,28)
print('输入数据',input_image.size())

my_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28*28, out_features=20),
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=10),
    nn.Softmax(dim=1)
)

# 单层信息 + 参数初始化
# 打印 layer 3
print("layer 3 structure: ", my_model[3])
for layer in my_model[3].modules(): #迭代器
    print(layer)
layer2 = list(my_model[3].modules())[0]
print(isinstance(layer, nn.Linear)) #判断layer是不是nn.Linear
print(isinstance(layer2, nn.Linear)) #判断layer是不是nn.Linear

# 打印 layer 3的所有参数
for name,param in my_model[3].named_parameters():
    print('layer3',name,'随机初始化',param.size(),'\n',param,'\n')

# 打印 layer 3的weight
print(my_model[3].weight.size())
print(layer.weight.size())
print(torch.equal(my_model[3].weight, layer.weight))

# 打印 layer 3的bias
print(my_model[3].bias.size())
print(layer.bias.size())
print(torch.equal(my_model[3].bias, layer.bias))

# 自定义初始化
def init_weight(tensor):
    with torch.no_grad():
        tensor *= (tensor.abs() >= 0.5).float() #绝对值小于0.5变成0
        # tensor = torch.cat((torch.eye(10),torch.eye(10)), dim = 1, out=tensor)
        return tensor

# # with torch.no_grad() 上下文管理器, wrap起来的语句将不会在反向转播中被记录
# aa = torch.tensor([1.0], requires_grad=True)
# bb = aa*2
# print(bb) # grad_fn=<MulBackward0>
# bb = aa+2
# print(bb) # grad_fn=<AddBackward0>
# with torch.no_grad():
#     cc = bb*2
#     print(cc) # 无grad_fn
# 参考：https://blog.csdn.net/weixin_46559271/article/details/105658654
    
# layer 3 参数初始化
for name,param in my_model[3].named_parameters():
    if 'weight' in name:
        init.normal_(param,mean=0,std=0.1)  #正态分布初始化
        print('layer3',name,'正态分布初始化',param.size(),'\n',param,'\n')
        init.uniform_(param,0,1)            #均匀分布初始化
        print('layer3',name,'均匀分布初始化',param.size(),'\n',param,'\n')
        init_weight(param)                  #自定义初始化
        print('layer3',name,'自定义初始化',param.size(),'\n',param,'\n')
    elif 'bias' in name:
        init.constant_(param,val=0)         #常数初始化
        print('layer3',name,'常数初始化',param.size(),'\n',param,'\n')

# 所有层信息 + 修改初始参数
# 打印 all layers
print("Model structure: ", my_model)
# print(list(my_model.modules()))
index = -1
for layer in my_model.modules(): #迭代器，第一个元素为Sequential，后续元素才是每一层模型
    print(index,layer)
    index += 1
    
# 打印 all layers 的参数名称
for name in my_model.state_dict():
    print('Layer:',name)

# 打印 all layers 的参数数值
for param in my_model.parameters():
    print(param,'\n')

# 打印 all layers 的参数名称及数值
# print(list(my_model.named_parameters()))
for name, param in my_model.named_parameters(): #迭代器generator
    print('Layer:',name,'初始参数',param.size(),'\n', param,'\n')
# print(my_model.state_dict())
for name in my_model.state_dict(): #有序字典OrderedDict
    print('Layer:',name,'初始参数','\n', my_model.state_dict()[name],'\n')

# 修改初始参数
for name,param in my_model.named_parameters():
    if 'weight' in name:
        init.constant_(param,val=1)         #常数初始化
        print('Layer:',name,'常数初始化',param.size(),'\n',param,'\n')
    elif 'bias' in name:
        init.constant_(param,val=0)         #常数初始化
        print('Layer:',name,'常数初始化',param.size(),'\n',param,'\n')

if torch.cuda.is_available():
    my_model.cuda()
    input_image = input_image.cuda()
output = my_model(input_image)

# 逐层分析
# 第0层 nn.Flatten (convert each 2D 28x28 image into a contiguous array of 784 pixel values)
layer0 = nn.Flatten() 
for name,param in layer0.named_parameters(): #无参数，所有没输出
    print('layer0',name,'随机初始化',param.size(),'\n',param,'\n')
temp = layer0(input_image)
print(temp.size())

# 第1层 线性层
layer1 = nn.Linear(in_features=28*28, out_features=20)
for name,param in layer1.named_parameters():
    print('layer1',name,'随机初始化',param.size(),'\n',param,'\n')
for name,param in layer1.named_parameters():
    if 'weight' in name:
        init.constant_(param,val=1)         #常数初始化
        print('layer1 weight 常数初始化',param.size(),'\n',param,'\n')
    elif 'bias' in name:
        init.constant_(param,val=0)         #常数初始化
        print('layer1 bias 常数初始化',param.size(),'\n',param,'\n')
temp = layer1(temp)
print('layer1 output:', temp.size(),'\n')

# 第2层 nn.ReLU (if x >= 0: return x; if x < 0: return 0)
# temp = nn.ReLU()(temp)
layer2 = nn.ReLU()
for name,param in layer2.named_parameters():
    print('layer2',name,'随机初始化',param.size(),'\n',param,'\n')
print('layer2 input:', temp.size(),'\n',temp,'\n') 
temp = layer2(temp)
print('layer2 output:', temp.size(),'\n',temp,'\n')

# 第3层 线性层
layer3 = nn.Linear(in_features=20, out_features=10)
for name,param in layer3.named_parameters():
    print('layer3',name,'随机初始化',param.size(),'\n',param,'\n')
for name,param in layer3.named_parameters():
    if 'weight' in name:
        init.constant_(param,val=1)         #常数初始化
        print('layer3 weight 常数初始化',param.size(),'\n',param,'\n')
    elif 'bias' in name:
        init.constant_(param,val=0)         #常数初始化
        print('layer3 bias 常数初始化',param.size(),'\n',param,'\n')
temp = layer3(temp)
print('layer3 output:', temp.size(),'\n')

# 第4层 nn.Softmax
layer4 = nn.Softmax(dim=1)
for name,param in layer4.named_parameters():
    print('layer4',name,'随机初始化',param.size(),'\n',param,'\n')
output2 = layer4(temp)
print('layer4 output:', output2.size())

print('检验nn.Sequential和逐层计算的结果是否相同:',torch.equal(output,output2))
#=============================================================================#
# 不同layer共享参数
linear10_10 = nn.Linear(in_features=10, out_features=10)
my_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28*28, out_features=20),
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=10),
    linear10_10, #layer 4 和 layer 5共享参数
    linear10_10,
    nn.Softmax(dim=1)
)
output = my_model(input_image)
# 打印 all layers
print("Model structure: ", my_model) 
# 打印 all layers 的参数名称及数值
for name, param in my_model.named_parameters():
    print('Layer:',name,'随机初始化',param.size(),'\n',param,'\n')
    #因为layer 4 和 layer 5共享参数，所以不输出layer 5的参数
#=============================================================================#
# 自动微分
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
# <=> w = torch.randn(5, 3)
# w.requires_grad_(True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
print(z.requires_grad)

#如果只需要进行forward computations，不需要tracking computations
with torch.no_grad():
    z2 = torch.matmul(x, w)+b
print(z2.requires_grad)
# or
z3 = z.detach()
print(z3.requires_grad)

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print('Gradient function for z =',z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

loss.backward()
print(w.grad) #叶节点，且requires_grad=True，有梯度
print(b.grad)
print(x.grad) #叶节点，且requires_grad=False，无梯度
print(z.grad) #不是叶节点，无梯度

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)

#注意，当我们第二次使用相同的参数向后调用时，渐变的值是不同的。 
# 发生这种情况是因为PyTorch在进行向后传播时会累积梯度，
# 即将计算出的梯度值添加到计算图所有叶节点的grad属性中。
# 如果要计算适当的梯度，则需要先将grad属性清零。
#=============================================================================#
# 优化模型参数（已整理至3_softmax.py中）
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)

learning_rate = 1e-3
batch_size = 64

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad() #reset the gradients of model parameters. Gradients by default add up; to prevent double-counting
        loss.backward()
        optimizer.step() #adjust the parameters by the gradients collected in the backward pass

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss() # Initialize the loss function
# Common loss functions include:
#     nn.MSELoss (Mean Square Error) for regression tasks
#     nn.NLLLoss (Negative Log Likelihood) for classification.
#     nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")
#=============================================================================#
# 保存和下载模型

# 只保存模型参数
torch.save(model.state_dict(), 'model_parameters.pth')

# 在加载参数之前，建立相同的模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork()

# 加载模型参数
model.load_state_dict(torch.load('model_parameters.pth'))
model.eval() #确保在推理之前执行，否则将产生不一致的结果

# 测试模型加载效果
X = torch.rand(1, 28, 28)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# 保存模型及参数
torch.save(model, 'model.pth')
# 加载模型及参数
model = torch.load('model.pth')
# 测试模型加载效果
X = torch.rand(1, 28, 28)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# import torch.onnx as onnx
# input_image = torch.zeros((1,3,224,224))
# onnx.export(model, input_image, 'model.onnx')


