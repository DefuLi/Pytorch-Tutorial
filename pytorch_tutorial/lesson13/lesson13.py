import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

# 计算信息熵
'''
p = torch.full([4], 1 / 4)  # 数据是4个，每个数据的概率是0.25
p_entropy = p * torch.log2(p)  # log是以2为底的
sum_entropy = -p_entropy.sum()
print(sum_entropy)  # tensor(2.)
'''

# 交叉熵
'''
x = torch.randn(1, 784)  # torch.Size([1,784])
w = torch.randn(10, 784)  # torch.Size([10,784])
logits = x @ w.t()  # x和w矩阵相乘
print(logits)  # torch.Size([1,10])
pred = F.softmax(logits, dim=1)
print(pred.shape)
print(pred[0].shape)
print(sum(pred[0]))
pred_log = torch.log(pred)
print(pred_log)
logits_nll_loss = F.nll_loss(pred_log, torch.tensor([3]))  # 自己手动计算cross entropy时需要将logits进行softmax和log计算，然后使用nll_loss函数
logits_cross_entropy = F.cross_entropy(logits, torch.tensor([3]))  # cross_entropy函数中已经有了softmax和log操作
print(logits_nll_loss)
print(logits_cross_entropy)
'''


# 多分类问题实战 共有10个输出

# 定义相关超参数
epoch = 5
batch_size = 200
learning_rate = 1e-3

# 下载手写数字数据集
# train=True代表卸载的训练部分 download=True代表默认的路径没有该数据集的话，可以在外部网站下载
# batch_size是说TPU一次处理图片的数量，并行处理，一次如果只处理一张图片，太耗时了。
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./dataset', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=batch_size, shuffle=True)

# 新建三个线性层
w1 = torch.randn(200, 784, requires_grad=True)  # 默认的第一个参数200是out 第二个参数784是in
b1 = torch.zeros(200, requires_grad=True)
w2 = torch.randn(200, 200, requires_grad=True)
b2 = torch.zeros(200, requires_grad=True)
w3 = torch.randn(10, 200, requires_grad=True)  # 输出节点有10个 因为是10分类问题
b3 = torch.zeros(10, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    x = x @ w1.t() + b1  # b1是[200] 会经过broadcast操作
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x


optimizer = torch.optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()  # torch.nn.CrossEntropyLoss函数和F.cross_entropy功能一样
for epoch in range(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        print('测试中。。。')
        print(logits.shape,target.shape)

        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        test_loss += criterion(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))
