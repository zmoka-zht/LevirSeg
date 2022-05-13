import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import torch.utils.data as data_utils
from PIL import Image

root_path = '/data02/zht_vqa/change_detection/LearnGroup/dataset/UCMerced'
data_file = '/data02/zht_vqa/change_detection/LearnGroup/dataset/UCMerced/train.txt'


#定义dataset 数据的读取,标签

class ClsData(data_utils.Dataset):
    def __init__(self, root_path, data_files, image_size=256):
        self.root = root_path
        self.data_files = np.loadtxt(data_files, dtype=np.str)
        self.class_list = os.listdir(
            os.path.join(self.root, 'train')
        )

        self.transformer = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor()
            ]
        )


    def __getitem__(self, item):
        #读取数据,数据的路径
        data_file = self.data_files[item]
        data_file = os.path.join(self.root, data_file)
        #print(img_path)
        data_file = data_file.replace('\\', '/')
        img = Image.open(data_file)

        #读取标签
        tmp = data_file.split('/')
        label_name = tmp[-2]
        label = self.class_list.index(label_name)

        #将Img和Label转换为pytorch框架可以操作的tensor
        img = self.transformer(img)
        label = torch.tensor(label)


        return img, label

    def __len__(self):
        return len(self.data_files)




if __name__ == '__main__':

    root_path = '/data02/zht_vqa/change_detection/LearnGroup/dataset/UCMerced'
    train_data_files = '/data02/zht_vqa/change_detection/LearnGroup/dataset/UCMerced/train.txt'
    test_data_files = '/data02/zht_vqa/change_detection/LearnGroup/dataset/UCMerced/test.txt'

    batch_size = 16

    train_dataset = ClsData(root_path, train_data_files)
    train_datalaoder = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = ClsData(root_path, test_data_files)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for i, data in enumerate(train_datalaoder):
        img, label = data
        print(img.shape)
        print(label.shape)

