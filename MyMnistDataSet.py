from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision
import torchvision.transforms as transforms

class MyMnistDataSet(Dataset):

    def __init__(self, root_dir, label_root_dir, type_name, transform=None): #取值为'train'或者'test'
        self.root_dir = root_dir
        self.type_name = type_name
        self.transform = transform
        self.path = os.path.join(self.root_dir, self.type_name)
        self.img_name_list = os.listdir(self.path) #得到path目录下所有图片名称的一个list

        self.label_root_dir = label_root_dir
        self.label_path = os.path.join(self.label_root_dir, self.type_name)
        self.label_name_list = os.listdir(self.label_path)

    def __getitem__(self, item):
        img_name = self.img_name_list[item]
        img_item_path = os.path.join(self.root_dir, self.type_name, img_name)
        img = Image.open(img_item_path)

        label_name = self.label_name_list[item]
        label_item_path = os.path.join(self.label_root_dir, self.type_name, label_name)
        label = Image.open(label_item_path)

        if self.transform is not None: #如果transform不等于None,那么执行转换
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.img_name_list)



"""
def main():
    dataset = MyMnistDataSet(root_dir='./mnist_dataset', type_name='train', transform=transforms.ToTensor())
    print(dataset[0])
    print(len(dataset))

"""

#if __name__ == "__main__":
#    #main()
#    main()
#    pass
