import torch
from torch import nn

m = nn.ConvTranspose2d(16, 33, 3, stride=2)

input = torch.randn(20, 16, 50, 100)

output = m(input)

print(output.shape)


# exact output size can be also specified as an argument
input = torch.randn(1, 16, 12, 12)

downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)

upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

h = downsample(input)
print(h.size())

output = upsample(h, output_size=input.size())
print(output.size())
