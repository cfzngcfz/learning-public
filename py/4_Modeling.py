# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:48:54 2021

@author: CC-i7-8750H
"""
# http://tangshusen.me/Dive-into-DL-PyTorch/#/chapter04_DL_computation/4.1_model-construction
import torch
from torch import nn
import torch.nn.init as init # 引入初始化模块

#-----------------------------------------------------------------------------#
# 1.建模
# 1.1.继承Module类构造模型
class NeuralNetwork(nn.Module):
    def __init__(self, **kwargs): #创建模型参数
        super(NeuralNetwork, self).__init__(**kwargs)
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 10)

    def forward(self, input): #定义前向计算
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        return input

model1 = NeuralNetwork()
print('1.Module类:', model1,'\n')
#-----------------------------------------------------------------------------#
# 1.2.Sequential类 - Module的子类
# 接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数，然后逐一添加Module的实例，模型的前向计算将这些实例按添加顺序依次计算
# 2)需按照顺序排列
# 3)保证相邻层的输入输出维度匹配
# 4)内部已实现forward功能
model2 = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 10),)
print('2.Sequential类:', model2,'\n')
# 或者写成逐层加入
model22 = nn.Sequential()
model22.add_module('linear', nn.Linear(784, 256))
model22.add_module('ReLU', nn.ReLU())
model22.add_module('2', nn.Linear(256, 10))
print('2.Sequential类:', model22,'\n')

# from collections import OrderedDict
# class MySequential(nn.Module): #功能与Sequential类相同
#     def __init__(self, *args):
#         super(MySequential, self).__init__()
#         if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
#             for key, module in args[0].items():
#                 self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
#         else:  # 传入的是一些Module
#             for idx, module in enumerate(args):
#                 self.add_module(str(idx), module)
#     def forward(self, input):
#         # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
#         for module in self._modules.values():
#             input = module(input)
#         return input
# model = MySequential(nn.Linear(784, 256),
#                      nn.ReLU(),
#                      nn.Linear(256, 10),)
# print('MySequential类:', model)
#-----------------------------------------------------------------------------#
# 1.3.ModuleList类 - Module的子类
# 1)仅是一个储存各种模块的列表，
# 2)模块之间没有顺序
# 3)可不用保证相邻层的输入输出维度匹配）
# 4)没有实现forward功能
class MyModuleList(nn.Module):
    def __init__(self):
        super(MyModuleList, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(784, 256)])
        self.layers.append(nn.ReLU())
        print('ModuleList[-1]:',self.layers[-1])
        self.layers.extend(nn.ModuleList([nn.Linear(256, 10)])) #在列表末尾一次性追加另一个列表
        print('ModuleList[-1]:',self.layers[-1])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
model3 = MyModuleList()
print('3.ModuleList类:', model3,'\n')
#-----------------------------------------------------------------------------#    
# 1.4.ModuleDict类 - Module的子类
# 与ModuleList相似，仅存储，无序，无forward
class MyModuleDict(nn.Module):
    def __init__(self):
        super(MyModuleDict, self).__init__()
        self.layers = nn.ModuleDict({'layer1': nn.Linear(784, 256),
                                     'layer2': nn.ReLU()})
        self.layers['layer3'] = nn.Linear(256, 10)
        print("ModuleDict['layer2']:",self.layers['layer2'])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
model4 = MyModuleDict()
print('4.ModuleDict类:', model4)
#-----------------------------------------------------------------------------#
# 1.5.多类嵌套

shared_layer = nn.Linear(3, 3) # Layer: 0.net.2 与 Layer: 1 共享参数

class layer_with_param(nn.Module): #自定义层
    def __init__(self):
        super(layer_with_param, self).__init__()
        """
        定义当前层的参数
        nn.Parameter(tensor_1) #一个参数
        nn.ParameterList([nn.Parameter(tensor_1),
                          nn.Parameter(tensor_2),
                          ...]) #参数列表
        nn.ParameterDict({'key1':nn.Parameter(tensor_1),
                          'key2':nn.Parameter(tensor_2),
                          ...}) #参数字典
        """
        # # nn.ParameterList 用法
        # self.params = nn.ParameterList([nn.Parameter(torch.randn(3, 3)) for _ in range(2)])
        # self.params.append(nn.Parameter(torch.randn(3, 2)))
        # self.params.extend([nn.Parameter(torch.randn(2, 1))])
        
        # nn.ParameterDict 用法
        self.params = nn.ParameterDict({
            'param1': nn.Parameter(torch.randn(3, 3)),
            'param2': nn.Parameter(torch.randn(3, 3)),
            'param3': nn.Parameter(torch.randn(3, 2))})
        self.params.update({'param4': nn.Parameter(torch.randn(2, 1))})
    def forward(self, x):
        # # nn.ParameterList 用法
        # for ii in range(len(self.params)):
        #     x = torch.mm(x, self.params[ii])
        
        # nn.ParameterDict 用法1
        for key in self.params.keys():
            x = torch.mm(x, self.params[key])

        # # nn.ParameterDict 用法2
        # for key,param in self.params.items():
        #     x = torch.mm(x, param)
        return x
    
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), shared_layer) #包含一个共享参数层

    def forward(self, x):
        return self.net(x)
model5 = NestMLP()
print("Model 5 structure:\n", model5)

model6 = nn.Sequential(NestMLP(), shared_layer, layer_with_param() ) #包含一个共享参数层，及自定义层
print("Model 6 structure:\n", model6)
print("Layer: 0.net.2 是否与 Layer: 1 共享参数?", id(model6[0].net[2])==id(model6[1]))
print(model6(torch.ones(1,4))) #测试模型
#-----------------------------------------------------------------------------#
# 2.参数
# 2.1.所有层参数
for name, param in model6.named_parameters(): #迭代器(generator),只输出共享参数首次，即Layer: 0.net.2的参数
    print('Layer:',name,'\n初始参数', param,'\n')

for name in model6.state_dict(): #有序字典OrderedDict，输出所有贡献参数，即Layer: 0.net.2 与 Layer: 1的参数
    print('Layer:',name,'\n初始参数', model6.state_dict()[name],'\n')

for param in model6.parameters(): #只输出共享参数首次
    print(param,'\n')

for index,layer in enumerate(model6.modules()): #输出模型结构
    print(index,layer)

# 2.2.单层参数
print('Sequential类第一层:\n', model6[0],'\n')
print('Module类net层:\n', model6[0].net,'\n')
print('model 6的第一层:\n', model6[0].net[0],'\n')
print('model 6的第一层的weight:\n', model6[0].net[0].weight,'\n')
print('model 6的第一层的weight的数值:\n', model6[0].net[0].weight.data,'\n')
print('model 6的第一层的weight的梯度:\n', model6[0].net[0].weight.grad,'\n') # 反向传播前梯度为None
print('model 6的第一层的bias:\n', model6[0].net[0].bias,'\n')


# 2.3.参数初始化

# 自定义初始化
def my_init_(param):
    with torch.no_grad():
        """
        # with torch.no_grad() 上下文管理器, wrap起来的语句将不会在反向转播中被记录
        aa = torch.tensor([1.0], requires_grad=True)
        bb = aa*2
        print(bb) # grad_fn=<MulBackward0>
        with torch.no_grad():
            cc = bb*2
            print(cc) # 无grad_fn
        参考：https://blog.csdn.net/weixin_46559271/article/details/105658654
        """
        param *= (param.abs() >= 0.5).float() #绝对值小于0.5变成0
        return param
    # # 或者 通过修改 param.data，不影响 param.grad
    # param.data += 1
    # return param
    
param = model6[0].net[0].weight
print('默认初始值:',param,'\n')
init.normal_(param,mean=0,std=0.1)  #正态分布初始化
print('正态分布初始化:',param,'\n')
init.constant_(param,val=0)         #常数初始化
print('常数初始化',param,'\n')
init.uniform_(param,0,1)            #均匀分布初始化
print('均匀分布初始化',param,'\n')
my_init_(param)                  #自定义初始化
print('自定义初始化',param,'\n')

# 2.4.参数梯度
linear = nn.Linear(1,1, bias=False)
model7 = nn.Sequential(nn.Linear(1,1, bias=False), linear, linear)
init.constant_(model7[0].weight, val=5)
init.constant_(model7[1].weight, val=3)
for name, param in model7.named_parameters():
    print('Layer:',name,'\n初始参数', param,'\n')
x = torch.tensor([[2.]])
y = model7(x).sum()

print('y = ',y) # y = weight2 * weight1 * weight0 * x = 3*3*5*2 = 90
y.backward(retain_graph=True)
print('dy/d(weight0) = ', model7[0].weight.grad) # dy/dweight0 = weight2 * weight1 * x = 3*3*2 = 18
print('dy/d(weight1) = ', model7[1].weight.grad) # 共享参数，梯度会叠加 dy/dweight1 + dy/dweight2 = weight2 * weight0 * x + weight1 * weight0 * x = 3*5*2+3*5*2 = 60

y.backward(retain_graph=True)
print('再次执行backward()后: dy/d(weight0) = ', model7[0].weight.grad) # 再次执行backward，梯度会叠加

model7[0].weight.grad.data.zero_() #梯度清零
y.backward(retain_graph=True)
print('梯度清零后: dy/d(weight0) = ', model7[0].weight.grad)
# 问题：2_3_gradinet 中的注意3和注意4没理解透
# 已发现: 1)修改x.data的值不影响对模型参数的求导，好像对自变量x的导数也不影响，需要深入研究
#     2) with torch.no_grad(): 在模型之前使用，不影响对参数求导，只影响对自变量求导
#-----------------------------------------------------------------------------#
# 读写 torch.save(obj) & torch.load(obj)

# 1. obj is a tensor, list of tensors, or dict of tensors
x = torch.ones(3)
y = torch.zeros(4)
torch.save(x, 'tensor_single.pt')
data1 = torch.load('tensor_single.pt')
print(data1)
torch.save([x,y], 'tensor_list.pt')
data2 = torch.load('tensor_list.pt')
print(data2)
torch.save({'x': x, 'y': y}, 'tensor_dict.pt')
data3 = torch.load('tensor_dict.pt')
print(data3)

# 2. obj is model with parameters
# 2.1.同时保存模型及参数
torch.save(model6, 'model_6_with_parameters.pth')
# 2.2.加载模型及参数
model6_load = torch.load('model_6_with_parameters.pth')
print(model6(torch.ones(1,4))) #测试模型

# 3.仅模型参数读写
# 3.1.保存模型参数
torch.save(model6.state_dict(), 'parameters_of_model_6.pth')
# 3.2.读取模型参数
# 在加载参数之前，建立相同的模型
model6_new = nn.Sequential(NestMLP(), shared_layer, layer_with_param() )
# 加载模型参数
model6_new.load_state_dict(torch.load('parameters_of_model_6.pth'))
print(model6_new(torch.ones(1,4))) #测试模型











