import torch
import numpy as np

'''
# 从np中创建，然后导入到tensor
a = np.array([2, 3.3])
print(a)
b = torch.from_numpy(a)
print(b)

c = np.ones([2, 3])
print(c)
d = torch.from_numpy(c)
print(d)
'''

'''
# 从tensor中直接创建 torch.tensor()小写的tensor中接收的是数据  torch.Tensor大写的Tensor中接收的是数据的维度  还有FloatTensor(shape)
a = torch.tensor([2., 3.2])
print(a)
b = torch.Tensor(2, 3)
print(b)
'''

'''
# 未初始化的数据(作者推荐初始化的数据 torch.rand randn等等) 一般数据随机 不规则 无穷大 无穷小 出现torch.nan 或 torch.inf错误 
a = torch.empty(2, 3)
print(a)
b = torch.FloatTensor(2, 3)
print(b)
c = torch.IntTensor(3, 3)
print(c)
'''

'''
# 生成数据 
a = torch.full([2, 3], 7)  # 两行三列中全为7
print(a)
b = torch.arange(0, 10)  # [0,1,2,...,9]
print(b)
c = torch.arange(0, 10, 2)  # 第三个参数是间隔
print(c)
d = torch.linspace(0, 10, steps=4)  # 第三个参数是数量
print(d)
e = torch.logspace(0, 1, steps=10)  # 10的x次方
print(e)
f = torch.ones(3, 3)
print(f)
g = torch.zeros(3, 3)
print(g)
h = torch.eye(3, 3)  # 单位阵 对角元素全为1
print(h)
'''

'''
# 随机打散
a = torch.randperm(10)
print(a)
idx = torch.randperm(2)  # [0,1]或[1,0]
print(idx)
'''

'''
# 索引和切片
a = torch.rand(4, 3, 28, 28)  # 4维
print(a[0].shape)  # torch.Size([3,28,28])
print(a[0, 0].shape)  # torch.Size([28,28])
print(a[0, 0, 2, 4])

print(a[:2].shape)  # torch.Size([2,3,28,28])
print(a[:2, 1:, :, :].shape)
print(a[:2, -1:, :, :].shape)  # torch.Size([2,1,28,28])
print(a[:2, -1, :, :].shape)  # torch.Size([2,28,28])

print(a[:, :, 0:28:2, 0:28:2].shape)  # torch.Size([4,3,14,14])
print(a[:, :, ::2, ::2].shape)  # 同上

print(a.index_select(0, torch.tensor([0, 2])).shape)  # 0指的是在第一个维度上进行操作 [0,2]需要转换为tensor 返回torch.Size([2,3,28,28])

print(a[...].shape)  # ...代表任意多的维度 返回torch.Size([4,3,28,28])
print(a[0, ...].shape)  # torch.Size([3,28,28])
print(a[..., :2].shape)  # torch.Size([4,3,28,2])
'''

'''
# 通过掩码mask 进行索引
a = torch.randn(3, 4)
print(a)
mask = a.ge(0.5)
print(mask)  # 大于等于0.5赋值为1，小于0.5赋值为0
b = torch.masked_select(a, mask)
print(b)  # a中大于等于0.5的元素

c = torch.tensor([[4, 3, 5], [6, 7, 8]])
print(c)
d = torch.take(c, torch.tensor([0, 2, 5]))  # take将c打平，返回tensor([4,5,8])
print(d)
'''