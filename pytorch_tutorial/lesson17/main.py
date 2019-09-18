import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim

from lenet5 import Lenet5


def main():
    epoch = 1000
    batch_size = 32

    cifar_train = datasets.CIFAR10('./datasets', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('./datasets', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    x, label = next(iter(cifar_train))
    print('x:', x.shape, 'label:', label.shape)

    # device = torch.device('cuda')  # 使用gpu计算
    model = Lenet5()
    print('model结构:', model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch_id in range(epoch):
        # train...
        model.train()
        for batch_id, (x, label) in enumerate(cifar_train):
            logits = model(x)  # logits[b,10]  label[b]
            loss = criterion(logits, label)  # logits和pred区别：pred是logits经过了softmax处理后的结果

            # 反向传播
            optimizer.zero_grad()  # 为什么要清零：每次反向传播时，不是重新写梯度，而是累加梯度
            loss.backward()
            optimizer.step()  # 更新了参数

        print(epoch_id, loss.item())

        # test...
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                logits = model(x)
                pred = logits.argmax(dim=1)  # 取最大值的索引 就是分类结果
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num  # 准确率
            print(epoch_id, acc)


if __name__ == '__main__':
    main()
