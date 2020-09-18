from UnetForFashionMnistNew import UNetForFashionMnistNew
import torch
from torch import nn
import time
import MyMnistDataSet
import sys
import torchvision.transforms as transforms


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    unet = UNetForFashionMnistNew(in_channel=1, out_channel=1)
    unet.load_state_dict(torch.load('./pretrained_models_noise/unet_model100.pth', map_location='cpu'))

    unet.eval()

    if use_cuda:
        unet.cuda()

    criterion = nn.MSELoss()

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    # 读取训练数据集
    batch_size = 512

    #准备数据
    mnist_test_dataset_with_noise = MyMnistDataSet.MyMnistDataSet(root_dir='./mnist_dataset_noise',
                                                                  label_root_dir='./mnist_dataset', type_name='test',
                                                                  transform=transforms.ToTensor())
    test_data_loader_with_noise = torch.utils.data.DataLoader(mnist_test_dataset_with_noise, batch_size, shuffle=False,
                                                              num_workers=num_workers)
    #遍历数据已有模型进行reference
    test_loss_sum, batch_count, start_time = 0.0, 0, time.time()
    for X, y in test_data_loader_with_noise:

        X = X.to(device)
        y = y.to(device)

        y_hat = unet(X)

        l = criterion(y_hat, y)

        test_loss_sum += l.cpu().item()
        batch_count += 1

    print('predict: batch_cout %d, test loss %.4f, time %.1f sec'
          % (batch_count, test_loss_sum / batch_count, time.time() - start_time))

if __name__=="__main__":
    main()

