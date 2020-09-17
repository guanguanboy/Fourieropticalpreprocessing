import torch
import torch.nn as nn
import d2lzh_pytorch as d2l
from torch.nn import init

class DNN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hiddens, num_outputs)
        self.flattern = d2l.FlattenLayer()

        for params in self.parameters():
            init.normal_(params, mean=0, std=0.01)  # 使用正态分布的方法初始化参数

    def forward(self, x):
        x = self.flattern(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x