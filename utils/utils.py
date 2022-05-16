try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch
import os

def save_checkpoint(ckpt_name:str, epoch_id: int, model_dict, path:str):
    torch.save({
        'epoch_id': epoch_id,
        'model_state_dict': model_dict,
    }, os.path.join(path, ckpt_name))

def load_checkpoint(ckpt_name:str, path:str, device:str, model):
    if os.path.exists(os.path.join(path, ckpt_name)):
        print("load last checkpoint")
        checkpoint = torch.load(os.path.join(path, ckpt_name), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("start from ", checkpoint['epoch_id'])
        return model
    else:
        print(" training from begin")