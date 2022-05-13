import torch
import torchvision
import torch.nn as nn

loss_dict = {'CrossEntropyLoss':nn.CrossEntropyLoss, 'L1Loss':nn.L1Loss}

def builder_loss(name:str='CrossEntropyLoss', **kwargs):
    '''

    :param name:
    :param kwargs:
    :return:
    '''
    if name in loss_dict.keys():
        return loss_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in available values.'.format(name))

if __name__=='__main__':
    loss = builder_loss()
    print(type(loss))