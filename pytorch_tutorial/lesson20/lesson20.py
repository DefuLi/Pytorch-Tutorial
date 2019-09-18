import numpy as np
import torch
from torch import nn, optim
from net import Net
from matplotlib import pyplot as plt

'''查看网络参数具体信息
model = Net()
for name, param in model.named_parameters():
    print(name, param.size())
'''

'''detach()作用是取出tensor的数据 并且另存到一个内存中 data是在原内存上操作
temp = torch.tensor([1., 2, 3], requires_grad=True)
print(id(temp))  # 查看内存 2063245745896
temp = temp.data 
print(id(temp))  # 2063245745896
temp = temp.detach()
print(id(temp))  # 2063245787352
'''

lr = 0.01
hidden_size = 16
num_time_steps = 50

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr)  # rnn.weight_ih_l0; rnn.weight_hh_l0; rnn.bias_ih_l0; rnn.bias_hh_l0; linear.weight; linear.bias

for iter in range(6000):
    hidden_prev = torch.zeros(1, 1, hidden_size)
    start = np.random.randint(3, size=1)[0]  # 返回0-3之间的一个整数
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)  # 目的是给定x波形 预测y波形 x和y波形相差一个单位
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)  # data的0-倒数第二个数
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)  # data的1-最后第一个数

    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()
    # 取出tensor数据 第一次反向传播后，会把计算图缓存清空，hidden_prev也会清空，
    # 那么第二次循环时，程序在model(x,hidden_prev)时会找不到hidden_prev。

    loss = criterion(output, y)  # 后面查看loss的数据类型
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print("Iteration:{} loss {}".format(iter, loss.item()))

# 预测阶段
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)  # 后面这个进行验证 去掉上面一行 data.shape=>(50) 不去的话(50,1) 验证成功

predictions = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    pred, hidden_prev = model(input, hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])  # ravel将多维数组转换为一维数组

x = x.data.numpy().ravel()  # 原数据
plt.scatter(time_steps[:-1], x, s=90)
plt.plot(time_steps[:-1], x)
plt.scatter(time_steps[1:], predictions)
plt.show()
