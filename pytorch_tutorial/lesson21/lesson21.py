import torch
from torch import nn

'''nn.LSTM使用方法
__init__：
input_size: The number of expected features in the input x
hidden_size: The number of features in the hidden state h(c和h尺寸一样)
num_layers: Number of recurrent layers

forward:
out,(ht,ct)=lstm(x,[ht_0,ct_0])
x: [seq,b,vec]
h/c: [num_layer,b,h]
out: [seq,b,h] 最后一层的所有时间戳的状态
'''

'''4层LSTM 基于nn.LSTM
lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
print(lstm)  # LSTM(100, 20, num_layers=4)
x = torch.randn(10, 3, 100)  # seq=10 b=3 vec=100
out, (h, c) = lstm(x)  # [h0,c0]省略

print(out.shape)  # out=>[10,3,20]
print(h.shape)  # h/c=>[4,b,20]
print(c.shape)
'''

'''nn.LSTMCell使用方法
__init__:
input_size: The number of expected features in the input x
hidden_size: The number of features in the hidden state h
num_layers: Number of recurrent layers

forward:
ht,ct=lstmcell(xt,[ht_0,ct_0]) xt=>[b,vec] ht/ct=>[b,h]
'''

'''单层LSTMCell(多层也有，视频课时94)
cell = nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)
x = torch.randn(10, 3, 100)
for xt in x:  # xt.shape=>[b,100]
    h, c = cell(xt, [h, c])

print(h.shape)  # torch.Size([3,20])
print(c.shape)  # torch.Size([3,20])
'''