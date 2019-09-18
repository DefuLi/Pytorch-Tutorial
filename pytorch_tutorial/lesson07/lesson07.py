import torch

# 合并与分割 cat stack split chunk

# cat没有创建新维度 stack创建了新维度 也就是新的概念
# split 按长度拆分 chunk按个数拆分
'''
a = torch.rand(4, 32, 8)  # 代表4个班级 每个班级32个学生 每个学生8门课成绩
b = torch.rand(5, 32, 8)  # 代表5个班级 每个班级32个学生 每个学生8门课成绩
c = torch.cat([a, b], dim=0)  # 将9个班级合并 torch.Size([9,32,8])
print(c.shape)
d = torch.rand(4, 3, 16, 32)
e = torch.rand(4, 3, 16, 32)
f = torch.cat([d, e], dim=2)  # torch.Size([4,3,32,32])
print(f.shape)
'''

'''
a = torch.rand(32, 8)
b = torch.rand(32, 8)
c = torch.stack([a, b], dim=0)  # stack用于将两个完全一样的tensor合并
print(c.shape)  # torch.Size([2,32,8]) 两个班级 每个班级32个人 每个人8门课成绩
d = torch.rand(32, 8)
e = torch.rand(32, 8)
f = torch.rand(32, 8)
g = torch.stack([d, e, f], dim=0)  # torch.Size([3,32,8])
print(g.shape)
h = torch.stack([d, e, f], dim=1)  # torch.Size([32,3,8])
print(h.shape)
'''

'''
a = torch.rand(2, 32, 8)
a1, a2 = a.split(1, dim=0)  # split by len 在0维上，拆分后每个是1
print(a1.shape, a2.shape)  # 两个torch.Size([1,32,8])
b = torch.rand(3, 32, 8)
b1, b2 = b.split([2, 1], dim=0)  # 在0维上，拆分后第一个是2，第二个是1
print(b1.shape, b2.shape)  # torch.Size([2,32,8])  torch.Size([1,32,8])

c = torch.rand(2, 32, 8)
c1, c2 = c.chunk(2, dim=0)  # chunk by num 在0维上，拆分成两个tensor
print(c1.shape, c2.shape)
'''