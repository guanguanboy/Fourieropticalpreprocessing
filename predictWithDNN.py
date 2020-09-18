import DNNModel
import torch
from torch import nn
import time
import MyMnistDataSet
import sys
import torchvision.transforms as transforms

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()

    num_inputs, num_outputs, num_hiddens = 784, 784, 50

    dnnModel = DNNModel.DNN(num_inputs=num_inputs, num_hiddens=num_hiddens, num_outputs=num_outputs)
    dnnModel.load_state_dict(torch.load('./pretrained_models_noise/dnn_model100.pth', map_location='cpu'))

    dnnModel.eval()

    if use_cuda:
        dnnModel.cuda()

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

        y = y.view(y.shape[0], -1)
        y = y.to(device)

        y_hat = dnnModel(X)

        l = criterion(y_hat, y)
        test_loss_sum += l.cpu().item()
        batch_count += 1

    print('predict: batch_cout %d, test loss %.4f, time %.1f sec'
          % (batch_count, test_loss_sum / batch_count, time.time() - start_time))




if __name__ == "__main__":
    main()


