import torch
import torch.nn as nn
from model.resnetfcn import ResNet50FCN
from model.cls_net import ClsNet
from model.seg_net import SegNet

#创建model字典
model_dict = {'cls_net':ClsNet,
              'seg_net':SegNet,
              'resnet50fcn':ResNet50FCN}

#创建build函数
def build_model(name:str='seg_net', **kwargs):
    if name in model_dict.keys():
        return model_dict[name](**kwargs)
    else:
        raise NotImplementedError("name not available values.".format(name))

if __name__=="__main__":
    model = build_model()
    print(model)