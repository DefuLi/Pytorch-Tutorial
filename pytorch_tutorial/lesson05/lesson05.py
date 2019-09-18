import torch

# 用的api有view reshape squeeze unsqueeze expand repeat transpose t permute

'''
# view和reshape功能一样，view是PyTorch0.3版本中的 维度的重新排列
a = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[9, 10, 11], [12, 13, 14]]])  # torch.Size([2,2,3])
print(a)
b = a.view(2, 2 * 3)
print(b.shape)

c = b.view(2, 2, 3)
print(c)

a = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[9, 10, 11], [12, 13, 14]]])  # torch.Size([2,2,3])
print(a)
b = a.reshape(2, 2 * 3)
print(b.shape)

c = b.reshape(2, 2, 3)
print(c)
'''

'''
# unsqueeze展开维度 squeeze挤压维度
a = torch.rand(4, 1, 28, 28)
print(a)
b = a.unsqueeze(0)  # 在0索引前面插入一个维度
print(b.shape)  # torch.Size([1,4,1,28,28])
c = a.unsqueeze(4)  # 在-1索引前面插入一个维度 具体的插入位置可以多次试验
print(c.shape)  # torch.Size([4,1,28,28,1])

d = torch.rand(32)  # 看成bias
e = torch.rand(4, 32, 14, 14)  # 目的是让d可以与e进行累加操作 前提是维度一致
d = d.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(d.shape)  # torch.Size([1,32,1,1])

f = d.squeeze()  # 默认挤压掉维度中size=1的所有维度
print(f.shape)  # torch.Size([32])

g = d.squeeze(1)  # 挤压掉dim=1这一维度 但是由于dim=1时，size≠1，所以无法挤压
print(g.shape)  # torch.Size([1,32,1,1])

h=d.squeeze(3)  # 挤压掉dim=3这一维度
print(h.shape)  # torch.Size([1,32,1])
'''

'''
# expand repeat维度扩展 不建议用repeat repeat拷贝张量数据 expand不会分配新的内存
a = torch.rand(4, 32, 14, 14)
b = torch.rand(32)
b = b.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # torch.Size([1,32,1,1])
print(b.shape)

c = b.expand(4, 32, 14, 14)  # 将1,31,1,1扩展到4,32,14,14  1->M可以（1->4）  M->N不可以（32->2不可以）
print(c.shape)

d = b.expand(-1, 32, -1, -1)  # 参数-1意味着不改变该维度 填-4是个bug
print(d.shape)  # torch.Size([1,32,1,1])

print(a.shape)  # torch.Size([4,32,14,14])
print(b.shape)  # torch.Size([1,32,1,1])
e = b.repeat(4, 32, 1, 1)  # repeat和expand功能相似 但repeat是参数和原参数相乘 expand是复制
print(e.shape)  # torch.Size([4,1024,1,1])
'''

'''
# 矩阵转置 t transpose
a = torch.rand(3, 4)
print(a.shape)
b = a.t()  # t函数转置 只能适用2D（2维）的tensor
print(b.shape)

c = torch.rand(4, 3, 28, 28)
d = c.transpose(1, 3)  # transpose交换1,3维度  只能一次两两交换
print(d.shape)  # 返回torch.Size([4,28,28,3])

e = torch.rand(4, 3, 28, 32)
f = e.transpose(1, 3).transpose(1, 2)
print(f.shape)  # torch.Size([4,28,32,3])
g = e.permute(0, 2, 3, 1)  # permute参数可以对多个维度进行重新排列
print(g.shape)  # torch.Size([4,28,32,3])
'''
