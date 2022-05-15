import argparse
import mmcv
from model.builder import build_model

parse = argparse.ArgumentParser()
parse.add_argument('--config_file',default=r'config/seg.py',type=str)


if __name__=='__main__':
    args =parse.parse_args()
    cfg = mmcv.Config.fromfile(args.config_file)
    #print(**cfg['config']['model_config'])
    model_cfg = cfg['config']['model_config']
    model = build_model(model_cfg['name'])
    print(model)