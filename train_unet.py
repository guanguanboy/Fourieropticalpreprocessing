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

import gc
from Unet import UNet
import torch
import sys
gc.collect()
use_gpu = torch.cuda.is_available()
from tqdm import tqdm, trange
import cv2
from UnetForFashionMnistNew import UNetForFashionMnistNew
import matplotlib.pyplot as plt
import time

#loss = nn.MSELoss()

#optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5

# 读取训练数据集
batch_size = 20


def resize(img, s=112):
    # print(s, img.shape)
    img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
    #label = cv2.resize(label, (s, s), interpolation=cv2.INTER_NEAREST)
    return img

def get_dataset():

    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

    train_set_size = len(mnist_train)
    test_set_size = len(mnist_test)
    print(train_set_size)
    print(test_set_size)

    train_feature, test_feature = [], []

    train_label, test_label = [], []

    for i in range(train_set_size):
        feature = mnist_train[i][0].numpy()
        train_feature.append(feature)
        train_label.append(mnist_train[i][0].numpy())

    for i in range(test_set_size):
        test_feature.append(mnist_test[i][0].numpy())
        test_label.append(mnist_test[i][0].numpy())

    print('train_feature size = ')
    print(len(train_feature))

    print('train_label size = ')
    print(len(train_feature))

    print(type(train_feature[0]))
    #print(train_feature[0].size()) # torch.Size([1, 28, 28])

    return np.array(train_feature), np.array(train_label), np.array(test_feature), np.array(test_label)

"""
mnist_train_data = mnist_loader.MnistForReconstruction(train_feature)
mnist_test_data = mnist_loader.MnistForReconstruction(test_feature)

train_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size, shuffle=False, num_workers=5)
train_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size, shuffle=False, num_workers=5)

print(mnist_train_data.__len__())
"""


def train_step(inputs, labels, optimizer, criterion, unet, width_out, height_out):
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = unet(inputs)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows)
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
    m = outputs.shape[0]
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = labels.resize(m*width_out*height_out)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss

def get_val_loss(x_val, y_val, width_out, height_out, unet):
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).long()
    if use_gpu:
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    m = x_val.shape[0]
    outputs = unet(x_val)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows)
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = y_val.resize(m*width_out*height_out)
    loss = F.cross_entropy(outputs, labels)
    return loss.data

def train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out):
    epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)
    t = trange(epochs, leave=True)
    for _ in t:
        total_loss = 0
        for i in range(epoch_iter):
            batch_train_x = torch.from_numpy(x_train[i * batch_size : (i + 1) * batch_size]).float()
            batch_train_y = torch.from_numpy(y_train[i * batch_size : (i + 1) * batch_size]).long()

            #我们这里输入直接是torch类型的list
            #batch_train_x = x_train[i * batch_size : (i + 1) * batch_size]
            #batch_train_y = y_train[i * batch_size : (i + 1) * batch_size]

            if use_gpu:
                batch_train_x = batch_train_x.cuda()
                batch_train_y = batch_train_y.cuda()
            batch_loss = train_step(batch_train_x , batch_train_y, optimizer, criterion, unet, width_out, height_out)
            total_loss += batch_loss
        if (_+1) % epoch_lapse == 0:
            val_loss = get_val_loss(x_val, y_val, width_out, height_out, unet)
            print("Total loss in epoch %f : %f and validation loss : %f" %(_+1, total_loss, val_loss))
    gc.collect()

def main():
    width_in = 28
    height_in = 112
    width_out = 28
    height_out = 28
    PATH = './unet.pt'
    #x_train, y_train, x_val, y_val = get_dataset(width_in, height_in, width_out, height_out)
    train_feature, train_label, test_feature, test_label = get_dataset()

    x_train = train_feature
    y_train = train_label
    x_val = test_feature
    y_val = test_label

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    batch_size = 3
    epochs = 1
    epoch_lapse = 50
    threshold = 0.5
    learning_rate = 0.01
    unet = UNet(in_channel=1,out_channel=2)
    if use_gpu:
        unet = unet.cuda()

    print(unet.eval())
    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(unet.parameters(), lr=0.5)

    train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, x_train, y_train, x_val,
          y_val, width_out, height_out)


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


def train():
    epoch = 100
    batch_size = 512


    #加载数据
    train_feature, train_label, test_feature, test_label = get_dataset()

    x_train = train_feature
    y_train = train_label
    x_val = test_feature
    y_val = test_label

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    unet = UNetForFashionMnistNew(in_channel=1,out_channel=1)
    if use_gpu:
        unet = unet.cuda()

    print(unet.eval())
    criterion = nn.MSELoss()

    shuffle = True

    optimizer = torch.optim.SGD(unet.parameters(), lr = 1e-2, momentum = 0.1)

    #epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)
    #t = trange(epochs, leave=True)
    #for _ in t:
    batch_cout = np.floor(x_train.shape[0] / batch_size).astype(int)
    for epo in range(0, epoch):
        start = time.time()
        for i in range(0, batch_cout):
            batch_train_x = torch.from_numpy(x_train[i * batch_size: (i + 1) * batch_size]).float()
            batch_train_y = torch.from_numpy(y_train[i * batch_size: (i + 1) * batch_size]).float()

            if use_gpu:
                batch_train_x = batch_train_x.cuda()
                batch_train_y = batch_train_y.cuda()

            optimizer.zero_grad()
            out = unet(batch_train_x)
            #print(type(out))
            #print(out.size())

            #X = []
            #for i in range(10):
            #    X.append(out[i])
            #show_fashion_mnist(X)

            #打印out和batch_train_x中前两个tensor的值

            if epo == 9 and i == 116:
                for i in range(3):
                    print('batch_train_x {i}:'.format(i=i))
                    print(batch_train_x[i])
                    print('out {i}:'.format(i=i))
                    print(out[i])

                X = []
                for i in range(10):
                    X.append(out[i])
                    show_fashion_mnist(X)
            """
            
            for i in range(3):
                print('batch_train_x {i}:'.format(i=i))
                print(batch_train_x[i])
                print('out {i}:'.format(i=i))
                print(out[i])
            """
            loss = criterion(out, batch_train_x)
            loss.backward()
            optimizer.step()

            print('This is batch {i} in epoch {epo}, the loss is {loss}'.format(i=i, epo=epo, loss=loss))
        print('epoch %d, time %.1f sec' % (epoch + 1, time.time() - start))


"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)
    if sys.argv[1] == 'train':
        train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out)
        pass
    else:
        if use_gpu:
            unet.load_state_dict(torch.load(PATH))
        else:
            unet.load_state_dict(torch.load(PATH, map_location='cpu'))
        print(unet.eval())
    #plot_examples(unet, x_train, y_train) #关注一下plot的实现
    #plot_examples(unet, x_val, y_val)
"""


if __name__ == "__main__":
    #main()
    train()
    pass