import torch
import numpy as np

# 标量和向量 张量数据类型 torch.randn torch.Size shape函数的使用

'''
# 数据类型检验
a = torch.randn(2, 3)  # 随机正态分布 新建2行3列 N(0,1)
print(a.type())  # tensor的类型
print(type(a))  # 用的少
print(isinstance(a, torch.FloatTensor))  # 参数的合法化检验
'''

'''
# 标量 Dimension=0
print(torch.tensor(1.))
print(torch.tensor(1.3))
a = torch.tensor(2.2)
print(a.shape)  # 返回torch.Size([])
print(len(a.shape))  # 返回0
print(a.size())  # 返回torch.Size([])

b = torch.tensor([2.2])
print(b.shape)  # 返回torch.Size([1])
print(len(b.shape))  # 返回1
print(b.size())  # 返回torch.Size([1])

c = torch.tensor([2, 3, 4, 5])
print(c.shape)  # 返回torch.Size([4])
print(len(c.shape))  # 返回1
print(c.size())  # 返回torch.Size([4])
'''

'''
# 向量 Dimension=1
a = torch.tensor([1.1, 2.2])  # .tensor接受的是数据的内容
print(a)

b = np.ones(2)
print(b)
c = torch.from_numpy(b)
print(c)  # tensor([1.,1.],torch.float64)
print(c.shape)
print(c.size())
'''

'''
# 向量 Dimension=2
a = torch.randn(2, 3)
print(a)
print(a.shape)
print(a.shape[0])  # 返回shape的第一个元素
print(a.size(1))  # 返回shape的第二个元素
'''

'''
# 向量 Dimension=3 经常在RNN中使用
a = torch.rand(1, 2, 3)  # 随机均匀分布 0-1之间 三维
print(a)
b = a.shape
print(b)
c = list(b)  # 使用list将torch.Size转化为list
print(c)
'''

'''
# 向量 Dimension=4 经常在CNN中使用
a = torch.rand(2, 3, 28, 28)  # 两张照片 三个通道 每个长宽为28*28
print(a)
b = a.shape
print(b)
print(b[2])
print(a.numel())  # 返回a的所有元素个数
print(a.dim())  # 返回a的总维数4
'''
