import torch
from torch.nn import functional as F

# 卷积神经网络 CNN
'''
# x=[b,1,28,28] k=[3,1,3,3] out=[b,3,26,26]
layer = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)  # 参数input channel 、kernel
x = torch.rand(1, 1, 28, 28)

out = layer.forward(x)
print(out.shape)  # torch.Size([1,3,26,26]) batch=1

layer = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer.forward(x)
print(out.shape)  # torch.Size([1,3,28,28])

layer = torch.nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
out = layer.forward(x)
print(out.shape)  # torch.Size([1,3,14,14])

# 推荐使用layer(x) 不推荐使用layer.forward(x)
out = layer(x)
print(out.shape)  # torch.Size([1,3,14,14])

parameters = layer.weight
print(parameters.shape)  # torch.Size([3,1,3,3])

bias = layer.bias
print(bias.shape)  # torch.Size([3])

# 或者使用F.conv2d(x,w,b,stride,padding)函数也可以完成卷积操作，但是使用Conv2d类更容易
'''

# 池化层 pooling
'''
out = torch.rand(1, 16, 14, 14)
x = out
layer = torch.nn.MaxPool2d(2, stride=2)  # 2为窗格大小 stride为步长
out = layer(x)
print(out.shape)  # torch.Size([1,16,7,7])

out = F.max_pool2d(x, 2, stride=2)  # 用F中的函数也可以
print(out.shape)  # torch.Size([1,16,7,7])
'''

# 向上采样 upsample
'''
out = torch.rand(1, 16, 14, 14)
x = out
out = F.interpolate(x, scale_factor=2, mode='nearest')  # scale_factor=2放大两倍 mode使用近邻差值
print(out.shape)  # torch.Size([1,16,28,28])
'''

# CNN relu函数
'''
x = torch.rand(1, 16, 7, 7)
layer = torch.nn.ReLU(inplace=True)
out = layer(x)
print(out.shape)  # torch.Size([1,16,7,7])

out = F.relu(x)
print(out.shape)  # torch.Size([1,16,7,7])
'''

# batch norm 批正则化 针对1d的数据
'''
x = torch.rand(100, 16, 784)  # 100张图片 16个通道 784代表28*28图片数据
layer = torch.nn.BatchNorm1d(16)  # 输入16个通道
out = layer(x)
print(out.shape)  # torch.Size([100,16,784])
mean = layer.running_mean
var = layer.running_var
print(mean)  # 本次运算统计出的mean和var
print(var)
'''

# batch norm 批正则化 针对2d的数据
'''
x = torch.rand(1, 16, 7, 7)  # 1张图片 16个通道 7*7图片尺寸
layer = torch.nn.BatchNorm2d(16)  # 16个通道 2维操作
out = layer(x)  # 进行平移和缩放操作 原先数据分布范围大 现在缩放到N(0,1)
print(vars(layer))
'''