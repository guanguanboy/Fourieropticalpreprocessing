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
import skimage
from skimage import io
from skimage import util
from PIL import Image
import numpy as np
import os

def get_dataset():

    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True)

    train_set_size = len(mnist_train)
    test_set_size = len(mnist_test)
    print(train_set_size)
    print(test_set_size)

    train_feature, test_feature = [], []

    train_label, test_label = [], []

    transform = transforms.Compose([transforms.ToTensor()])
    unloader = transforms.ToPILImage()

    for i in range(train_set_size):
        img = transform(mnist_train[i][0]).numpy()
        noisy = skimage.util.random_noise(img, mode='gaussian', var=0.01)
        #imgTensor = transform(noisy) #将numpy转换为tensor
        #imgTensor = imgTensor.squeeze(0)
        #imgPIL = unloader(imgTensor, type='uint8')#将tensor转换为PIL
        imgPIL = unloader(noisy.astype(np.uint8))
        imgPIL.save('./mnist_dataset_noise/train/{i}.png'.format(i=i+1))

    for i in range(test_set_size):
        img = transform(mnist_train[i][0]).numpy() #PIL转换为numpy:方式：先转换为Tensor再转换为numpy
        noisy = skimage.util.random_noise(img, mode='gaussian', var=0.01) #numpy 加噪声
        #imgTensor = transform(noisy) #numpy转换为tensor
        #imgTensor = imgTensor.squeeze(0)
        #imgPIL = unloader(imgTensor)#将tensor转换为PIL
        imgPIL = unloader(noisy.astype(np.uint8))
        imgPIL.save('./mnist_dataset_noise/test/{i}.png'.format(i=i+1))

def get_dataset2():
    img_name_list = os.listdir('./mnist_dataset/train/')  # 得到path目录下所有图片名称的一个list

    for i in range(len(img_name_list)):
        img_name = img_name_list[i]
        img_item_path = os.path.join('./mnist_dataset/train/', img_name)

        img = skimage.io.imread(img_item_path)
        noisy = skimage.util.random_noise(img, mode='gaussian', var=0.01) #numpy 加噪声
        img_item_save_path = os.path.join('./mnist_dataset_noise/train/', img_name)
        noisy_255 = (noisy*255).astype(np.uint8);
        skimage.io.imsave(img_item_save_path, noisy_255)


def get_dataset3():
    img_name_list = os.listdir('./mnist_dataset/test/')  # 得到path目录下所有图片名称的一个list

    for i in range(len(img_name_list)):
        img_name = img_name_list[i]
        img_item_path = os.path.join('./mnist_dataset/test/', img_name)

        img = skimage.io.imread(img_item_path)
        noisy = skimage.util.random_noise(img, mode='gaussian', var=0.01) #numpy 加噪声
        img_item_save_path = os.path.join('./mnist_dataset_noise/test/', img_name)
        noisy_255 = (noisy*255).astype(np.uint8);
        skimage.io.imsave(img_item_save_path, noisy_255)


if __name__ == "__main__":
    #main()
    get_dataset2()
    get_dataset3()
    pass