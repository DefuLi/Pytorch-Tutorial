import torch
from torch.nn import functional as F

# 全连接层

# 没有使用激活函数
'''
x = torch.randn(1, 784)  # torch.Size([1,784]) 假设做为输入的数据
layer1 = torch.nn.Linear(784, 200)  # 784是in，200是out
layer2 = torch.nn.Linear(200, 200)
layer3 = torch.nn.Linear(200, 10)
x = layer1(x)  # torch.Size([1,200])
x = layer2(x)  # torch.Size([1,200])
x = layer3(x)  # torch.Size([1,10])
print(x.shape)
'''

# relu激活函数
'''
x = torch.randn(1, 784)
layer1 = torch.nn.Linear(784, 200)
layer2 = torch.nn.Linear(200, 200)
layer3 = torch.nn.Linear(200, 10)
x = layer1(x)
x = F.relu(x, inplace=True)  # inplace=True代表会进行覆盖运算 节省内存
x = layer2(x)
x = F.relu(x, inplace=True)
x = layer3(x)
x = F.relu(x, inplace=True)
print(x.shape)  # torch.Size([1,10])
'''


# 自定义Module需要注意
# Linear继承nn.Module
# 需要对Linear进行初始化
# 需要实现forward()函数 不需要实现backward()函数
'''
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 200),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 10),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x
'''
