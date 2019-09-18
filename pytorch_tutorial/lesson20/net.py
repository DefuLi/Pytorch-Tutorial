import torch
from torch import nn

input_size = 1
hidden_size = 16
output_size = 1
num_layers = 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(  # RNN类的初始化 nn.RNN(1,16,1)
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # True，保证输入的x第一个参数是batch
        )

        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)  # 权值的初始化 后期可以更改

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev
