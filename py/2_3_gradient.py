# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:23:39 2021

@author: CC-i7-8750H
"""

# torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
import torch
tensorX = torch.ones(2, 2)
print('Function:',tensorX.grad_fn,'; is leaf node?',tensorX.is_leaf,'; requires grad?', tensorX.requires_grad)
tensorX.requires_grad_(True)
print('Function:',tensorX.grad_fn,'; is leaf node?',tensorX.is_leaf,'; requires grad?', tensorX.requires_grad)
tensorX = torch.ones(2, 2, requires_grad=True)
print('Function:',tensorX.grad_fn,'; is leaf node?',tensorX.is_leaf,'; requires grad?', tensorX.requires_grad)
tensorY = tensorX + 2
print('Function:',tensorY.grad_fn,'; is leaf node?',tensorY.is_leaf,'; requires grad?', tensorY.requires_grad)

tensorZ = tensorY * tensorY * 3
out = tensorZ.mean()

# 注意1：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，
# 梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。
"""
out.backward() # <=> out.backward(torch.tensor(1.))
程序在执行backward后，会释放计算图的缓存
如果需要再次执行，需用out.backward(retain_graph=True)
"""
# tensorX.grad.data.zero_() #梯度清零
out.backward(retain_graph=True) #保留计算图的缓存
print(tensorX.grad) #标量，即d(out)/dx，如果反向传播执行多次，grad会累加
#-----------------------------------------------------------------------------#
# 注意2：张量对张量求导的理解
tensorX = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
tensorY = 2 * tensorX
tensorZ = tensorY.view(2, 2)
print(tensorZ) #张量

tensorW= torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
tensorZ.backward(tensorW)
# tensorW是和tensorZ同形的张量，则tensorZ.backward(tensorW)的含义是：
# 先计算l = torch.sum(tensorZ * tensorW)，则l是个标量，然后再求l关于自变量tensorX的导数
# 进而保证tensorX.grad是和tensorX同形的张量
print(tensorX.grad)
#-----------------------------------------------------------------------------#
# 注意3：中断梯度追踪
tensorX = torch.tensor(1.0, requires_grad=True)
tensorY1 = tensorX ** 2 
with torch.no_grad():
    tensorY2 = tensorX ** 3
tensorZ = tensorY1 + tensorY2

print(tensorX.requires_grad)
print(tensorY1, tensorY1.requires_grad) # True
print(tensorY2, tensorY2.requires_grad) # False
print(tensorZ, tensorZ.requires_grad) # True

tensorZ.backward()
print(tensorX.grad) # z=x**2在x=1处的导数，而不是z=x**2+x**3在x=1处的导数，因为y2被包裹，不反向转播给x
#-----------------------------------------------------------------------------#
# 注意4：如果只想修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么可以对tensor.data进行操作
# 暂时没想到这样做的意义在于何处？
tensorX = torch.ones(1,requires_grad=True)
print(tensorX, tensorX.requires_grad)
print(tensorX.data, tensorX.data.requires_grad) # 还是一个tensor,但是已经是独立于计算图之外
tensorX.data *= 100    # 只改变了值，不会记录在计算图，所以不会影响梯度传播
tensorY = 2 * tensorX
tensorY.backward()
print(tensorX)        # 更改data的值也会影响tensor的值
print(tensorX.grad)




