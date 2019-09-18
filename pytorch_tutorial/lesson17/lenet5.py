import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(Lenet5, self).__init__()

        # 卷积层 初始化
        self.conv_unit = nn.Sequential(
            # x:[b,3,32,32]=>[b,6,]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # pooling层不改变channel的数量，只改变长宽
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        # 全连接层 初始化
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        '''
        tmp = torch.randn(2, 3, 32, 32)  # [b,3,32,32]
        out = self.conv_unit(tmp)
        print('conv out:', out.shape)  # 卷积层输出 torch.Size([2,16,5,5])
        '''

        # use Cross Entropy Loss
        self.criterion = nn.CrossEntropyLoss()  # criterion评价标准的意思 nn.CrossEntropyLoss是个类 需要先初始化

    def forward(self, x):
        """
        :param x: [b,3,32,32]
        :return:
        """
        batch_size = x.size(0)  # x=[b,3,32,32]=>b
        x = self.conv_unit(x)  # [b,3,32,32]=>[b,16,5,5]
        x = x.view(batch_size, 16 * 5 * 5)  # [b,16,5,5]=>[b,16*5*5]
        logits = self.fc_unit(x)  # tensor.Size([b,10])

        return logits


def main():
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet out:', out.shape)


if __name__ == '__main__':
    main()
