import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 单层感知机 perceptron 输入的x特征有10个 权值w也有10个
'''
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)
o = torch.sigmoid(x @ w.t())  # x与w矩阵相乘 x[1,10] w[1,10] 所以w需要转置 得到o为一个数
print(o.shape)  # o.shape=torch.Size([1,1])
loss = F.mse_loss(torch.ones(1, 1), o)  # 假设label是torch.ones(1,1)
print(loss)
print(loss.shape)  # torch.Size([])
loss.backward()
print(w.grad)  # 有loss对10个w值的导数值 也就是偏导数
'''

# 多层感知机（多输出的，没有隐藏层） x还是10个，w是[2,10]代表输出层只有两个神经元
'''
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
o = torch.sigmoid(x @ w.t())  # 矩阵相乘 就是神经网络中的第一层求和 [1,10]*[10,2]=[1,2]
print(o.shape)
loss = F.mse_loss(torch.ones(1, 2), o)  # 原视频中是torch.ones(1,1)，这样也行的原因是程序使用了broadcast机制
print(loss)
loss.backward()
print(w.grad)
print(w.shape)  # totch.Size([2,10])
'''

# 链式法则 y1=x*w1+b1 y2=y1*w2+b2 有一层的隐藏层
'''
x = torch.tensor(1.)  # x.shape=tensor.Size([]) 标量
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(1.)  # tensor(*.)点的作用是声明为浮点型类型 只有浮点型类型才能被求导
w2 = torch.tensor(2., requires_grad=True)
b2 = torch.tensor(1.)
y1 = x * w1 + b1  # 1*2+1=3
y2 = y1 * w2 + b2  # 3*2+1=7
dy2_dy1 = torch.autograd.grad(y2, [y1], retain_graph=True)[0]  # grad得到的是一个tuple
dy1_dw1 = torch.autograd.grad(y1, [w1], retain_graph=True)[0]
dy2_dw1 = dy2_dy1 * dy1_dw1  # 链式法则结果tensor(2.) 等于dy2_dw1=torch.autograd.grad(y2,[w1],retain_graph=True)[0]
print(dy2_dw1)
'''


# 2D函数优化实例 函数Himmelblau function f(x,y)=(x**2+y-11)**2+(x+y**2-7)**2
'''
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)  # X Y分别是两个相同尺寸的矩阵 X和Y相同位置，共同代表一组x,y坐标值
print(X.shape, Y.shape)  # (120,120)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x_y = torch.tensor([0., 0.], requires_grad=True)  # 本次优化的是dpred/dx 而不是derror/dw
optimizer = torch.optim.Adam([x_y], lr=1e-3)  # 自动完成x'=x-0.001▽x  y'=y-0.001▽y
for step in range(20000):
    pred = himmelblau(x_y)
    optimizer.zero_grad()  # 梯度清零
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print('step{}: x={}, f(x)={}'.format(step, x_y.tolist(), pred.item()))
'''
