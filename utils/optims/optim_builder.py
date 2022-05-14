import torch.optim as optim
import torch

#创建optims的字典
optims_dict = {'SGD':optim.SGD,
               'Adam':optim.Adam}

#创建bulider函数实例化optim
def build_optim(name:str='SGD', **kwargs):
    '''

    :param name:
    :param kwargs:
    :return:
    '''
    if name in optims_dict.keys():
        return optims_dict[name](**kwargs)
    else:
        raise NotImplementedError("name %s not available values.".format(name))

if __name__=='__main__':

    opt = build_optim('SGD')
    print(opt)