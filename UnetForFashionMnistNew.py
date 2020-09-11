import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


# Down sampling module
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU()
    )


# Up sampling module
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )

class UNetForFashionMnistNew(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNetForFashionMnistNew, self).__init__()
        self.conv1 = add_conv_stage(in_channel, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)

        self.conv2m = add_conv_stage(128, 64)
        self.conv1m = add_conv_stage(64, 32)

        self.conv0 = nn.Sequential(  #把conv0搞清楚
            nn.Conv2d(32, out_channel, 3, 1, 1),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        # Encode
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))

        conv3m_out_ = torch.cat((self.upsample32(conv3_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)
        conv0_out = self.conv0(conv1m_out)

        return conv0_out