import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import torch.utils.data as data_utils
from PIL import Image


root_path = '/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen'
data_files = '/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen/train.txt'

class SegData(data_utils.Dataset):
    def __init__(self, root_path, data_files, image_size=256):
        self.root = root_path
        self.data_files = np.loadtxt(data_files, dtype=np.str)

        self.transformer = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor()
            ]
        )

    def __getitem__(self, item):
        #图像怎么获取,标签图像怎么获取
        data_file = self.data_files[item]
        data_file = os.path.join(self.root, data_file)
        data_file = data_file.replace('\\', '/')

        img = Image.open(data_file)

        #标签图像获取
        label_file = data_file.replace('tif', 'png')
        label = Image.open(label_file)

        #转换成tensor
        img = self.transformer(img)
        label = torch.from_numpy(np.array(label)).long()

        return img, label



    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    root_path = '/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen'
    train_data_files = '/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen/train.txt'
    val_data_files = '/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen/valid.txt'
    test_data_files = '/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen/test.txt'
    batch_size = 16


    train_dataset = SegData(root_path, train_data_files)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = SegData(root_path, val_data_files)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = SegData(root_path, test_data_files)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for i, data in enumerate(train_dataloader):
        img, label = data
        print(img.shape)
        print(label.shape)