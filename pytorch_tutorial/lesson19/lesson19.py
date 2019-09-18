import torch
from torch import nn
from torchnlp.word_to_vector import GloVe

'''
vectors = GloVe(name='6B', dim=100)
vectors = GloVe()
print(vectors['hello'])
'''

'''
rnn = nn.RNN(100, 10)  # 100=>word dim 10=>mem size
print(rnn._parameters.keys())  # odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'])
print(rnn.parameters())
print(rnn.weight_ih_l0.shape)  # torch.Size([10,100])
print(rnn.weight_hh_l0.shape)  # torch.Size([10,10])

for p in rnn.parameters():
    print(p.type())
'''

'''
nn.RNN使用方法
__init__:三个参数 input_size hidden_size num_layers
input_size=>word dim. The number of expected features in the input x
hidden_size=>mem size. The number of features in the hidden state h
num_layers=>默认是1. Number of recurrent layers

out,ht=forward(x,h0) x=>[5,3,100] 5个单词 3句话 每个单词100维表示； h0=>[layer=1,3,10] 层数默认1 3句话 10mem size
x:[seq len,b,word vec]
h0/ht:[num layers,b,h dim]
out:[seq len,b,h dim] ht相当于最后一个mem,out是5个mem 但是是最后一层的
'''

''' 单层的RNN
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)  # 单词维度100 mem维度20  mem和后面的h是一个意思
x = torch.randn(10, 3, 100)  # x=>[10,3,100] batch为3 seq len为10 word vec为100
out, h = rnn.forward(x, torch.zeros(1, 3, 20))  # h0=>num layers=1; batch=3; h dim=20
print(out.shape)  # torch.Size([10,3,20]) 10个时刻的输出
print(h.shape)  # torch.Size([1,3,20]) 最后一个时刻的输出
'''

'''两层的RNN
rnn = nn.RNN(input_size=100, hidden_size=10, num_layers=2)
print(rnn._parameters.keys())  # 两层RNN所有的参数
print(rnn.weight_ih_l0.shape)  # torch.Size([10,100])
print(rnn.weight_ih_l1.shape)  # torch.Size([10,10]) 第二层的输入就是第一层的mem
'''

'''四层的RNN
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
x = torch.randn(10, 3, 100)
out, h = rnn.forward(x)
print(out.shape)  # torch.Size([10,3,20])  10个单词=>最后一层的状态
print(h.shape)  # torch.Size([4,3,20])  最后一个时间戳上所有的状态
'''

'''调用RNNCell实现RNN
cell1 = nn.RNNCell(100, 20)
h0 = torch.randn(3, 20)
x0 = torch.randn(3, 100)
h1 = cell1(x0, h0)
print(h1.shape)  # torch.Size([3,20])
'''

