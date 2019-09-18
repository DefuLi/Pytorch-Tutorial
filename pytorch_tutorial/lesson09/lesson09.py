import torch

# norm-p p范数
'''
a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(a)
print(b)
print(c)
a_norm_1 = a.norm(1)  # a,b,c的1范数
b_norm_1 = b.norm(1)
c_norm_1 = c.norm(1)
print(a_norm_1)  # tensor(8.)
print(b_norm_1)  # tensor(8.)
print(c_norm_1)  # tensor(8.)

a_norm_2 = a.norm(2)  # a,b,c的2范数
b_norm_2 = b.norm(2)
c_norm_2 = c.norm(2)
print(a_norm_2)  # tensor(2.8284)  根号8
print(b_norm_2)  # tensor(2.8284)  根号8
print(c_norm_2)  # tensor(2.8284)  根号8

b_norm_1_0 = b.norm(1, dim=0)  # 1范数 针对b.shape=[2,4]中的2，返回tensor([2,2,2,2])
print(b_norm_1_0)
b_norm_1_1 = b.norm(1, dim=1)  # 1范数 针对b.shape=[2,4]中的4，返回的tensor([4,4])
print(b_norm_1_1)

c_norm_1_0 = c.norm(1, dim=0)  # 1范数 针对dim=0，针对哪个维度，哪个维度就会消掉
print(c_norm_1_0)  # c_norm_1_0.shape=[2,2]
'''

# 数据统计量 mean sum min max prod
'''
a = torch.arange(8).view(2, 4).float()  # 生成shape为[2,4],类型是float的1-8数据
print(a.dtype)
print(a)
print(a.min())  # 最小值
print(a.max())  # 最大值
print(a.mean())  # 平均值
print(a.prod())  # 累乘
print(a.sum())  # 求和
print(a.argmax())  # 最大值的索引值 所有的[2,4]经过了打平，返回了7

b = torch.randn(4, 10)  # 理解为4张照片，每张照片属于0-9类的概率 我们想要的是返回每张照片属于哪个类
print(b.argmax())  # 返回tensor(31) 不是我们想要的，因为程序默认打平了4*10个索引
print(b.argmax(dim=1))  # tensor([0,1,5,7]) 每张照片中概率最大的索引值
'''

# dim和keepdim的作用
'''
a = torch.randn(4, 10)
a_max = a.max()  # 默认将4*10数据打平 然后找到最大值
print(a_max)
a_max_1 = a.max(dim=1)  # dim=1 返回4行中的最大值，以及索引值
print(a_max_1)

a_max_keepdim=a.max(dim=1,keepdim=True)  # keepdim是为了返回的结果的dim和原tensor a的dim一致 原a的dim为2，如果不加keepdim，返回结果dim=1
print(a_max_keepdim[0].shape)  # tensor.Size([4,1]) 没有keepdim的话，tensor.Size([4])

a_argmax=a.argmax(dim=1,keepdim=True)
print(a_argmax.shape)  # torch.Size([4,1])
'''

# topk kthvalue topk可以返回比max更多的信息
'''
a = torch.randn(4, 10)
print(a.shape)
a_topk = a.topk(3, dim=1)  # 返回每一行中最大的前三个数据 以及索引值
print(a_topk)

a_topk_false = a.topk(3, dim=1, largest=False)  # 默认largest=True 设置为False时返回的最小的3个
print(a_topk_false)

a_kthvalue = a.kthvalue(8, dim=1)  # 在dim=1的维度上，返回第8个小的数和索引值
print(a_kthvalue)
'''

# 数据比较 eq(a,b) equal(a,a) ga(a,0)
'''
a = torch.randn(4, 10)
print(a > 0)
print(torch.gt(a, 0))  # 效果是和a>0一致 可类比到> >= < <= != ==

b = torch.ones(2, 3)
c = torch.randn(2, 3)
b_c_eq = torch.eq(b, c)  # 判断b,c对应位置元素是否相等 相等的元素按位置返回1
print(b_c_eq)
b_c_equal = torch.equal(b, c)  # 判断b,c整体是否相等，相等返回True
print(b_c_equal)
a_a_equal = torch.equal(a, a)  # 返回True
print(a_a_equal)
'''