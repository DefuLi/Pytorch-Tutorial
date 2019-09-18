import torch

# broadcast 自动扩展成一致的维度  不拷贝数据 不占用内存
# [4,32,8]+5.0在数学上是没有办法直接相加的 使用broadcast,程序自动将5.0这个数扩展成[4,32,8]的维度，并进行相加
# 小维度指定 大维度随意 [4,32,8]例如代表4class 32studen 8score 5.0为score 这是所有8门课程都加5分
# 也可以在8门课中只对一门课加5分

# broadcast 不需要写代码 是程序的默认规则 默认实现了若干个unsqueeze和expand操作 目的是让两个Tensor的shape一致


