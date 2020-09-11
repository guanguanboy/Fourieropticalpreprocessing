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

def get_dataset():

    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True)

    train_set_size = len(mnist_train)
    test_set_size = len(mnist_test)
    print(train_set_size)
    print(test_set_size)

    train_feature, test_feature = [], []

    train_label, test_label = [], []

    for i in range(train_set_size):
        mnist_train[i][0].save('./mnist_dataset/train/{i}.png'.format(i=i+1))

    for i in range(test_set_size):
        mnist_test[i][0].save('./mnist_dataset/test/{i}.png'.format(i=i+1))

if __name__ == "__main__":
    #main()
    get_dataset()
    pass