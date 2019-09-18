import torch
from torch.nn import functional as F

# sigmoid函数
'''
a = torch.linspace(-100, 100, 10)
print(a)
b = torch.sigmoid(a)

print(b)
'''

# tanh函数
'''
a = torch.linspace(-10, 10, 10)
print(a)
b = torch.tanh(a)
print(b)
'''

# relu函数
'''
a = torch.linspace(-1, 1, 10)
print(a)
b = torch.relu(a)
c = F.relu(a)  # F中也有relu函数，同时也有sigmoid和tanh函数
print(b)
print(c)
'''

# autograd.grad 自动求导
# pred=xw+b
'''
# 定义w时直接标注为requires_grad=True
x = torch.ones(1)  # x=tensor([1.])
w = torch.full([1], 2, requires_grad=True)  # w=tensor([2.])
mse = F.mse_loss(torch.ones(1), x * w)  # y=torch.ones(1)=1 x*w=2 mse=(1-2)**2=1
print(mse)  # tensor(1.)

mse_w_grad = torch.autograd.grad(mse, [w])  # 对w进行求导时，需要先把w进行标注，标注为requires_grad=True
print(mse_w_grad)
'''

'''
# 后期更新w.requires_grad_() 因为定义w时没有标注为requires_grad=True
x = torch.ones(1)  # x=tensor([1.])
w = torch.full([1], 2)  # w=tensor([2.])
mse = F.mse_loss(torch.ones(1), x * w)  # y=torch.ones(1)=1 x*w=2 mse=(1-2)**2=1
print(mse)  # tensor(1.)

w.requires_grad_()  # 后期标注w为可求导

mse = F.mse_loss(torch.ones(1), x * w)  # 重新更新图 这是动态图 w更新后，需要把图也更新
mse_w_grad = torch.autograd.grad(mse, [w])  # 对w进行求导时，需要先把w进行标注，标注为requires_grad=True
print(mse_w_grad)
'''

# 使用loss.backward函数进行求导
'''
x = torch.ones(1)
w = torch.full([1], 2, requires_grad=True)
mse = F.mse_loss(torch.ones(1), x * w)
mse.backward()  # backward自动从后往前传播 自动计算所有requires_grad=True的变量导数值
print(w.grad)  # tensor([2.])
'''

# F.softmax函数
'''
a = torch.rand(3, requires_grad=True)  # 对a进行了标注 后面会对a进行求导
p = F.softmax(a, dim=0)  # softmax函数
print(p)
print(sum(p))  # tensor(1.)

p1_a_grad = torch.autograd.grad(p[1], [a], retain_graph=True)
print(p1_a_grad)  # p[1]对[a]中三个数求导，retain_graph=True代表了下次可以继续求导

p2_a_grad = torch.autograd.grad(p[2], [a])  # p[2]对[a]中三个数求导
print(p2_a_grad)
'''
