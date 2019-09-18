import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from matplotlib import pyplot as plt
from utils import plot_curve, plot_image, one_hot

# load dataset
batch_size = 512

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])
                               ),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])
                               ),
    batch_size=batch_size, shuffle=False
)
x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(28 * 28, 256)  # 三层网络 Linear两个参数分别为输入和输出维数
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x:[b,1,28,28]
        # h1=relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2=relu(h1*w2+b2)
        x = F.relu(self.fc2(x))
        # h3=h2*w3+b3
        x = self.fc3(x)

        return x


net = Net()
# [w1,b1,w2,b2,w3,b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        # x:[b,1,28,28],y:[512]
        # [b,1,28,28]=> [b,784]
        x = x.view(x.size(0), 28 * 28)
        # => [b,10]
        out = net(x)

        y_onehot = one_hot(y)
        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out:[b,10]=> pred:[b]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
