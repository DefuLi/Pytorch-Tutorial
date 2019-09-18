import torch

# torch高级操作OP where gather where根据条件 gather查表
# where三个参数 where(condition.x,y)
'''
cond = torch.ByteTensor([[1, 1], [1, 0]])
print(cond.shape)
a = torch.tensor([[1, 2], [3, 4]])
print(a.shape)
b = torch.tensor([[5, 6], [7, 8]])
print(b.shape)

a_b_where = torch.where(cond, a, b)  # tensor([[1,2],[3,8]]) cond中是条件，为1的地方代表a中同位置元素
print(a_b_where)  # torch.where可以用GPU高度并行运算 自己用python写的话是CPU运行慢

# gather是查表操作 就是映射
prob = torch.rand(4, 10)  # 假设prob是概率值 4张照片属于0-9十个类别的概率
idx = prob.topk(dim=1, k=3)  # 每张照片概率排名前三的数值
print(idx)
idx_1 = idx[1]  # idx的索引值
print(idx_1)
label = torch.arange(10) + 100  # tensor([100,101,102,103,104,,,109])
print(label)
idx_gather = torch.gather(label.expand(4, 10), dim=1, index=idx_1.long())  # label.expand是一张表 index是查表的索引
print(idx_gather)  # 返回的就是idx_1中对应映射到label.expand中的值 tensor.Size([4,3])
'''
