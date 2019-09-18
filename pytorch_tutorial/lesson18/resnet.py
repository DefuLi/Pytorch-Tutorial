import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)  # ch_in图片ch数量 ch_out卷积核数量
        self.bn1 = nn.BatchNorm2d(ch_out)  # batchnorm2d的第一个参数为输入到bn1类中的特征数，也就是conv1的输出ch_out
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),  # [b,ch_in,h,w]=>[b,ch_out,h,w]
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 输入x经卷积和正则化后进行relu
        out = self.bn2(self.conv2(out))  # 输入的out经卷积和正则化后输出赋值给out
        # 进行shortcut: x+out，但是x和out的维度不一定一致，进行逐元素求和会报错误 所以需要进行维度变换
        # self.extra: [b,ch_in,h,w]=>[b,ch_out,h,w]
        out = self.extra(x) + out

        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(  # 预处理层
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

        #  接下来是4个block
        self.block1 = ResBlock(64, 128, stride=2)  # [b,64,h,w]=>[b,128,h,w]
        self.block2 = ResBlock(128, 256, stride=2)  # [b,128,h,w]=>[b,256,h,w]
        self.block3 = ResBlock(256, 512, stride=2)  # [b,256,h,w]=>[b,512,h,w]
        self.block4 = ResBlock(512, 512, stride=2)  # [b,512,h,w]=>[b,1024,h,w]

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # print('after conv:', x.shape)  # [b,512,2,2]
        x = F.adaptive_avg_pool2d(x, [1, 1])  # [b,512,2,2]=>[b,512,1,1] 任意的输入 都可以转变为[1,1],这是参数可以更改
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        # print('after view: ',x.shape)
        x = self.outlayer(x)

        return x


def main():
    blk = ResBlock(64, 128, stride=2)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block:', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)


if __name__ == '__main__':
    main()
