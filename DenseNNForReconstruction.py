import torch
from torch import nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms

import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import mnist_loader

print(torch.__version__)

#定义模型
num_inputs, num_outputs, num_hiddens = 784, 10, 784

net = nn.Sequential(
    nn.Linear(num_inputs, num_hiddens), # nn.Linear就是一个全连接层
    nn.ReLU(),
    nn.Linear(num_hiddens, num_inputs)
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01) #使用正态分布的方法初始化参数

loss = nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5

# 读取训练数据集
batch_size = 20

# 获取原始数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

train_set_size = len(mnist_train)
test_set_size = len(mnist_test)

train_feature, test_feature = [], []

train_label, test_label = [], []

for i in range(train_set_size):
    train_feature.append(mnist_train[i][0])
    train_label.append(mnist_train[i][0])

for i in range(test_set_size):
    test_feature.append(mnist_test[i][0])
    test_label.append(mnist_test[i][0])

mnist_train_data = mnist_loader.MnistForReconstruction(train_feature)
mnist_test_data = mnist_loader.MnistForReconstruction(test_feature)

train_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size, shuffle=False,
                                           num_works=5)

#需要从原始数据集中构造出Y与X，然后返回合适的train_iter和test_iter


#d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
