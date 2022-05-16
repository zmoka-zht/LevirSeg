import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import torch
from torchvision.utils import make_grid
from dataset.seg_dataset import SegData
import torch.utils.data as data_utils
from model.seg_net import SegNetz

def vis_acc_curve(root_path:str):
    train_mf1 = np.load(os.path.join(root_path, 'train_mf1.npy'))
    val_mf1 = np.load(os.path.join(root_path, 'val_mf1.npy'))
    n = len(val_mf1)
    plt.plot(range(1, n+1), train_mf1[0:n])
    plt.plot(range(1, n+1), val_mf1[0:n])
    plt.legend(['train_mf1', 'val_mf1'])
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('mean F1')
    plt.savefig(os.path.join(root_path, 'acc.png'))

def vis_train_loss_curve(root_path:str):
    train_loss = np.load(os.path.join(root_path, 'train_loss.npy'))
    n = len(train_loss) #500
    plt.plot(range(1, n + 1), train_loss[0:n])
    plt.legend(['train_loss'])
    plt.grid(True)
    plt.xlabel('iter')
    plt.ylabel('train loss')
    plt.savefig(os.path.join(root_path, 'train_loss.png'))

def vis_val_loss_curve(root_path:str):
    val_loss = np.load(os.path.join(root_path, 'val_loss.npy'))
    n = len(val_loss) #500
    plt.plot(range(1, n + 1), val_loss[0:n])
    plt.legend(['val_loss'])
    plt.grid(True)
    plt.xlabel('iter')
    plt.ylabel('val loss')
    plt.savefig(os.path.join(root_path, 'val_loss.png'))


def inv_normalize_image(data):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * rgb_mean + rgb_std
    return data.clip(0, 1)

def label2image(prelabel, colormap):
    h, w = prelabel.shape
    prelabel = prelabel.reshape(h*w, -1)
    image = np.zeros((h*w, 3), dtype='int32')
    for i in range(len(colormap)):
        index = np.where(prelabel == i) # 标签对应的colormap索引
        image[index, :] = colormap[i]
    return image.reshape(h, w, 3)

def vis(dataloader, device, model, colormap):
    for i, data in enumerate(dataloader):
        if i > 0:
            break

        test_img, test_label = data
        test_img = test_img.to(device)
        test_label = test_label.to(device)

        test_logit, test_prob = model(test_img)

        test_preb = torch.argmax(test_prob, dim=1)

        test_img_numpy = test_img.cpu().numpy()
        test_img_numpy = test_img_numpy.transpose(0, 2, 3, 1)
        test_label_numpy = test_label.cpu().numpy()
        test_preb_numpy = test_preb.cpu().data.numpy()

        plt.figure(figsize=(16, 10))
        for i in range(4):
            plt.subplot(3, 4, i + 1)
            plt.imshow(inv_normalize_image(test_img_numpy[i]))
            plt.axis('off')
            plt.title(str(i + 1))
            plt.subplot(3, 4, i + 5)
            plt.imshow(label2image(test_label_numpy[i], colormap))
            plt.axis('off')
            plt.title(str(i + 5))
            plt.subplot(3, 4, i + 9)
            plt.imshow(label2image(test_preb_numpy[i], colormap))
            plt.axis('off')
            plt.title(str(i + 9))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()


if __name__=="__main__":
    #root_path = r'/home/Wuwei/WuweiStuH/PycharmProjects/LevirSeg/checkpoint/seg'
    root_path = r'/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen'
    val_data_files = r'/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen/valid.txt'
    model_pth = r'/home/Wuwei/WuweiStuH/PycharmProjects/LevirSeg/checkpoint/seg/best_ckpt.pt'
    device = 'cuda:0'
    #vis_acc_curve(root_path)
    #vis_val_loss_curve(root_path)

    val_dataset = SegData(root_path, val_data_files)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)
    model = SegNetz().to(device)
    checkpoint =torch.load(model_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    colormap = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                [255, 255, 0], [255, 0, 0]]

    vis(dataloader=val_dataloader, model=model, device=device, colormap=colormap)

