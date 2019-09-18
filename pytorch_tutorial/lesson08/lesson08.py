import torch

# tensor基本数学运算
# torch.sub=-  torch.mul=*点乘  torch.div=/
# //是整除的意思
''' 减法
a = torch.rand(3, 4)
b = torch.rand(4)
ab_add1 = a + b  # 程序使用了broadcast机制 自动扩展了b的维度
ab_add2 = torch.add(a, b)  # 运算符+和add一样
eq = torch.eq(ab_add1, ab_add2)
eq[1, 3] = torch.tensor(0)
print(eq)
print(torch.all(eq))  # tensor(0,dtype=torch.uint8) 因为我把eq[1,3]的1改为了0 所以torch.all(eq)为0
'''

# 矩阵相乘（需满足i*j j*k的要求) 推荐matmul  mm matmul @都是对等的 但是mm只针对二维
'''
a = torch.tensor([[3, 3], [3, 3]],
                 dtype=torch.float)  # torch.Size([2,2]) 也可以用torch.Tensro默认dtype是float的 或者[3.,3.][3.,3.]也可以识别为float
print(a.dtype)
b = torch.ones(2, 2)
print(b)
c = torch.mm(a, b)
print(c)
d = torch.matmul(a, b)
print(d)
e = a @ b
print(e)
'''

# 次方运算 power
'''
a = torch.full([2, 2], 3)  # 2行2列全为3
print(a)
a_pow2 = a.pow(2)  # pow(2)平方 等于**
print(a_pow2)
a_sqrt = a_pow2.sqrt()  # 平方根
print(a_sqrt)
'''

# e次方 log运算
'''
a = torch.ones(2, 2)
print(a.shape)
print(a.dtype)
a_exp = torch.exp(a)
print(a_exp)
a_log=torch.log(a)
print(a_log)
'''

# 近似值 floor ceil round trunc frac
# floor下 ceil上 trunc截取整数 frac截取小数
'''
a = torch.tensor(3.14)
print(a.floor())  # tensor(3.)
print(a.ceil())  # tensor(4.)
print(a.trunc())  # tensor(3.)
print(a.frac())  # tensor(0.1400)

b = torch.tensor(3.499)  # 四舍五入 小于3.5 返回3
print(b.round())  # tensor(3.)
'''

# 梯度裁剪 gradient clipping
'''
grad = torch.rand(2, 3)
grad = grad * 15
print(grad)
# print(grad.max())  # 最大值
# print(grad.median())  # 中间值
grad_clamp1 = grad.clamp(10)  # 小于10的数会被替换成10
print(grad_clamp1)
grad_clamp2 = grad.clamp(0, 10)  # 元素在0-10之间的保留，其余的替换为10
print(grad_clamp2)
'''