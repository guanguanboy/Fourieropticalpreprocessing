import torch
from torch.utils import data

class MnistForReconstruction(data.Dataset):

    def __init__(self, feature):
        self.feature = feature
        self.label = feature

    def __len__(self):
        len(self.feature)

    def __getitem__(self, index):
        return self.feature[index], self.label[index]