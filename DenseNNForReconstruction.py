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
import MyMnistDataSet
import time
import matplotlib.pyplot as plt

print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#定义模型
num_inputs, num_outputs, num_hiddens = 784, 784, 50

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens), # nn.Linear就是一个全连接层
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs)
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01) #使用正态分布的方法初始化参数

loss = nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 100

# 读取训练数据集
batch_size = 512

# 获取原始数据集
#需要从原始数据集中构造出Y与X，然后返回合适的train_iter和test_iter
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4

mnist_train_dataset = MyMnistDataSet.MyMnistDataSet(root_dir='./mnist_dataset', label_root_dir='./mnist_dataset', type_name='train', transform=transforms.ToTensor())
train_data_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size, shuffle=False,
                                                num_workers=num_workers)

mnist_test_dataset = MyMnistDataSet.MyMnistDataSet(root_dir='./mnist_dataset', label_root_dir='./mnist_dataset', type_name='test', transform=transforms.ToTensor())
test_data_loader = torch.utils.data.DataLoader(mnist_test_dataset, batch_size, shuffle=False,
                                               num_workers=num_workers)

mnist_train_dataset_with_noise = MyMnistDataSet.MyMnistDataSet(root_dir='./mnist_dataset_noise', label_root_dir='./mnist_dataset', type_name='train', transform=transforms.ToTensor())
train_data_loader_with_noise = torch.utils.data.DataLoader(mnist_train_dataset, batch_size, shuffle=False,
                                                num_workers=num_workers)

mnist_test_dataset_with_noise = MyMnistDataSet.MyMnistDataSet(root_dir='./mnist_dataset_noise', label_root_dir='./mnist_dataset', type_name='test', transform=transforms.ToTensor())
test_data_loader_with_noise = torch.utils.data.DataLoader(mnist_test_dataset, batch_size, shuffle=False,
                                               num_workers=num_workers)

#训练网络

#该函数无法使用，因为我们做的是图像重建，所以没法直接用重建后的像素值是否相等来比较准确率，只能通过MSE或者SSIM来衡量
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X in data_iter:

            y = X
            y = y.view(y.shape[0], -1)

            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == (y.to(device)).argmax(dim=1)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y.argmax(dim=1)).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y.argmax(dim=1)).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def show_fashion_mnist(images):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    figs = plt.subplots(1, len(images), figsize=(12, 12), sharey=True)
    i = 1
    for image in images:
        plt.subplot(1, 10, i)
        img = image.cpu()
        #fig = figs[i]
        print(type(img))
        plt.imshow(img.view((28, 28)).detach().numpy())
        #plt.axes.set_title(lbl)
        #plt.axes.get_xaxis().set_visible(False)
        #plt.axes.get_yaxis().set_visible(False)
        i = i + 1
    plt.show()

def train(net, train_iter, test_iter, loss, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X,y in train_iter:

            X = X.to(device)
            y = y.view(y.shape[0], -1)
            #print('y.shape = ')
            #print(y.shape)
            y = y.to(device)

            y_hat = net(X)
            #print('y_hat.shape = ')
            #print(y_hat.shape)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            #train_acc_sum += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

            if epoch == 99 and batch_count == 110:
                X = []
                for i in range(10):
                    X.append(y_hat[i])
                show_fashion_mnist(X)

            #print('batch count %d' % batch_count)

        #test_acc = evaluate_accuracy(test_iter, net)

        #print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
        #      % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        print('epoch %d, loss %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, time.time() - start))

print(net)
train(net, train_data_loader, test_data_loader, loss, batch_size, optimizer, device, num_epochs)

#train(net, train_data_loader_with_noise, test_data_loader_with_noise, loss, batch_size, optimizer, device, num_epochs)