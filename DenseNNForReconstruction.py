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
import DNNModel

print(torch.__version__)



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

def evaluateDNN(net, testDataLoader, epoch, loss, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    test_loss_sum, batch_count, start_time = 0.0, 0, time.time()
    with torch.no_grad():
        for X, y in testDataLoader:
            if isinstance(net, torch.nn.Module):
                net.eval() #进入评估模式，这会关闭dropout等
                X = X.to(device)

                y = y.view(y.shape[0], -1)
                y = y.to(device)

                y_hat = net(X)

                l = loss(y_hat, y)
                test_loss_sum += l.cpu().item()
                batch_count += 1

                #改回训练模式
                net.train()

        print('epoch %d, batch_cout %d, test loss %.4f, time %.1f sec'
              % (epoch + 1, batch_count, test_loss_sum / batch_count, time.time() - start_time))

def train(net, train_iter, test_iter, loss, batch_size, optimizer, device, num_epochs, is_noise_data=False):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X,y in train_iter:

            #print('X.shape =')
            #print(X.shape)
            X = X.to(device)

            #print('original y.shape = ')
            #print(y.shape)
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

        if (epoch + 1) % 10 == 0:
            if (is_noise_data):
                torch.save(net.state_dict(),
                           './pretrained_models_noise/dnn_model%d.pth' % (epoch + 1))  # save for every 10 epochs
            else:
                torch.save(net.state_dict(),
                           './pretrained_models/dnn_model%d.pth' % (epoch + 1))  # save for every 10 epochs
            #print('batch count %d' % batch_count)

        #test_acc = evaluate_accuracy(test_iter, net)

        #print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
        #      % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        print('epoch %d, loss %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, time.time() - start))

        evaluateDNN(net, test_iter, epoch, loss, device)


def trainDenseNN():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义模型
    num_inputs, num_outputs, num_hiddens = 784, 784, 50

    net = DNNModel.DNN(num_inputs=num_inputs, num_hiddens=num_hiddens, num_outputs=num_outputs)

    """
    net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),  # nn.Linear就是一个全连接层
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs)
    )

    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)  # 使用正态分布的方法初始化参数
    """

    loss = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

    num_epochs = 100

    # 读取训练数据集
    batch_size = 512

    # 获取原始数据集
    # 需要从原始数据集中构造出Y与X，然后返回合适的train_iter和test_iter
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    mnist_train_dataset = MyMnistDataSet.MyMnistDataSet(root_dir='D:/DataSets/mnist_dataset', label_root_dir='D:/DataSets/mnist_dataset',
                                                        type_name='train', transform=transforms.ToTensor())
    train_data_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size, shuffle=False,
                                                    num_workers=num_workers)

    mnist_test_dataset = MyMnistDataSet.MyMnistDataSet(root_dir='D:/DataSets/mnist_dataset', label_root_dir='D:/DataSets/mnist_dataset',
                                                       type_name='test', transform=transforms.ToTensor())
    test_data_loader = torch.utils.data.DataLoader(mnist_test_dataset, batch_size, shuffle=False,
                                                   num_workers=num_workers)

    mnist_train_dataset_with_noise = MyMnistDataSet.MyMnistDataSet(root_dir='D:/DataSets/mnist_dataset_noise',
                                                                   label_root_dir='D:/DataSets/mnist_dataset', type_name='train',
                                                                   transform=transforms.ToTensor())
    train_data_loader_with_noise = torch.utils.data.DataLoader(mnist_train_dataset_with_noise, batch_size,
                                                               shuffle=False,
                                                               num_workers=num_workers)

    mnist_test_dataset_with_noise = MyMnistDataSet.MyMnistDataSet(root_dir='D:/DataSets/mnist_dataset_noise',
                                                                  label_root_dir='D:/DataSets/mnist_dataset', type_name='test',
                                                                  transform=transforms.ToTensor())
    test_data_loader_with_noise = torch.utils.data.DataLoader(mnist_test_dataset_with_noise, batch_size, shuffle=False,
                                                              num_workers=num_workers)

    print(net)
    train(net, train_data_loader, test_data_loader, loss, batch_size, optimizer, device, num_epochs)
    #train(net, train_data_loader_with_noise, test_data_loader_with_noise, loss, batch_size, optimizer, device, num_epochs, True)

if __name__ == "__main__":
    trainDenseNN()