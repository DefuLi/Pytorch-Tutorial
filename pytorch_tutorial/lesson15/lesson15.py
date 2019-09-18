import torch
from torch.nn import functional as F
from visdom import Visdom
from torchvision import datasets, transforms

# 计算准确率
'''
logits = torch.randn(4, 10)
pred = F.softmax(logits, dim=1)
pred_label = pred.argmax(dim=1)
print(pred_label)  # tensor([4,0,7,6])

label = torch.tensor([4, 0, 7, 5])
correct = torch.eq(pred_label, label)  # tensor([0,0,0,0], dtype=torch.uint8)
print(correct)

accuracy = correct.sum().float().item() / 4  # float()变为floattensor类型 item()是转为numpy值
print(accuracy)
'''

# Visdom可视化
'''
# 安装visdom 并且开启服务
viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
'''

# train validation test集的划分
'''
batch_size = 512

# train_db共有60kb的样本，test_db共有10kb的样本。需要将train_db的样本重新划分为50kb和10kb，划分为validation集
train_db = datasets.MNIST('./dataset', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))  # (0.1307,)逗号的作用是括号中只有一个元素的时候，将该元素转为元组类型。
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)

test_db = datasets.MNIST('./dataset', train=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Normalize((0.1307,),
                                                                                                      (0.3081,))]))
test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)
print('train:', len(train_db), 'test:', len(test_db))

train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])  # random_split函数用来划分数据集
print('train:', len(train_db), 'val:', len(val_db))

train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_db, batch_size=batch_size, shuffle=True)
'''

# 正则化 regularization 防止过拟合
'''
parameters = torch.randn(1, 20)
optimizer = torch.optim.SGD(parameters, lr=1e-3, weight_decay=0.01)  # weight_decay参数就是L2正则化的lambda
'''

# 动量 momentum 更有效的梯度下降
'''
parameters = torch.randn(1, 20)
optimizer = torch.optim.SGD(parameters, lr=1e-3, momentum=0.78, weight_decay=0.01)  # momentum是动量
# Adam优化器已经内置了momentum参数，不需要手动设置。只有SGD优化器需要手动设置
'''

# 学习率衰减 scheme1
'''
parameters = torch.randn(1, 20)
optimizer = torch.optim.SGD(parameters, lr=1e-3, momentum=0.78, weight_decay=0.01)  # momentum是动量

# 耐心值设置为10，下面调用10次loss还没有减少时，学习率会乘factor，进行减少。
# 耐心值监听的是scheduler.step(loss)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1,
                                                       patience=10)
'''

# 学习率衰减 scheme2
'''
parameters = torch.randn(1, 20)
optimizer = torch.optim.SGD(parameters, lr=1e-3, momentum=0.78, weight_decay=0.01)  # momentum是动量
# step_size指的是epoch或者其他循环中变量，30个epoch后，lr乘以0.1 scheduler.step()放在epoch循环中进行监听
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
'''

# Dropout是有一定概率断开神经元之间连接
'''
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(784, 200),
    torch.nn.Dropout(0.5),  # 不是在全连接层，是隐藏层的输出到隐藏层的输入，直连接，一对一。
    torch.nn.ReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 10),
)
# 在train的时候，使用dropout，在test的时候不能使用
epochs=10
for epoch in range(epochs):
    # train ...
    net_dropped.eval()  # 测试时不使用
    # test ...
'''
