from dataset.seg_dataset import SegData
from dataset.cls_dataset import ClsData
from losses.builder import builder_loss
from metric.caculate_metric import segmengtion_metric
from model.builder import build_model
from utils.optims.optim_builder import build_optim
import mmcv
import torch.utils.data as data_utils
import torch
import numpy as np
import torch.nn as nn
import os
from utils.utils import save_checkpoint, load_checkpoint


class SegInterface(object):
    def __init__(self, args):
        cfg = mmcv.Config.fromfile(args.config_file)
        self.model_cfg = cfg['config']['model_config']
        self.train_cfg = cfg['config']['train_cfig']
        self.num_workers = cfg['config']['train_cfig']['num_workers']
        self.epoch_mf1 = 0.
        self.mode = build_model(self.model_cfg['name'])
        self.best_val_mf1 = 0.
        self.best_epoch_id =0
        self.load_epoch_id = 0


    def tain(self):
        #断点载入
        if os.path.exists(os.path.join(self.train_cfg['checkpoint_path'],'last_ckpt.pt')):
            print("load last checkpoint")
            checkpoint = torch.load(os.path.join(self.train_cfg['checkpoint_path'], 'last_ckpt.pt'),
                                    map_location=self.train_cfg['device'])
            self.mode.load_state_dict(checkpoint['model_state_dict'])
            self.mode.to(self.train_cfg['device'])
            self.load_epoch_id = checkpoint['epoch_id']
            print("start from ", checkpoint['epoch_id'])

        #初始化训练所需参数
        batch_size = self.train_cfg['batch_size']
        #max_epoch = self.train_cfg['num_epoch']
        lr = self.train_cfg['lr']
        device = self.train_cfg['device']
        checkpoint_path=self.train_cfg['checkpoint_path']
        log_path = self.train_cfg['log_path']

        #初始化数据集
        train_dataset = SegData(self.train_cfg['train_data']['root_path'],
                                 self.train_cfg['train_data']['train_data_files'])
        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=self.num_workers)

        val_dataset = SegData(self.train_cfg['train_data']['root_path'],
                                self.train_cfg['train_data']['val_data_files'])
        val_dataloader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=self.num_workers)

        #初始化损失、优化器及分割评价
        criterion = builder_loss(self.train_cfg['criterion'])
        optimer = build_optim(self.train_cfg['optimer'], params=self.mode.parameters(), lr=lr)
        self.mode.to(device)
        metric = segmengtion_metric(self.model_cfg['out_channel'], device)


        #开始训练
        for epoch_id in range(self.load_epoch_id, self.train_cfg['num_epoch']):
            print("......Begin Training......")
            metric.clear()
            self.mode.train()

            for batch_id, data in enumerate(train_dataloader):
                train_img, train_label = data
                train_img = train_img.to(device)
                train_label = train_label.to(device)

                train_logits, train_prob = self.mode(train_img)
                train_loss =criterion(train_logits, train_label)

                optimer.zero_grad()
                train_loss.backward()
                optimer.step()

                #计算每个batch的精度
                label = train_label.detach().squeeze()
                pred = torch.argmax(train_prob.detach(), dim=1)
                metric.update_confusion_matrix(gt=label, pred=pred)
                m_train =len(train_dataloader)
                if np.mod(batch_id, 20) == 1:
                    score_dict_batch = metric.get_matrix_per_batch(gt=label, pred=pred)
                    message = 'Training: [%d,%d][%d,%d], train_loss: %.5f\n' % (epoch_id, self.train_cfg['num_epoch']-1, batch_id, m_train, train_loss.item())
                    message_class_list = ['{key}: {value:.5f} '.format(key=key, value=value.item()) for key, value in score_dict_batch.items()]
                    message_class = ''.join(map(str, message_class_list)) + '\n'
                    print(message + message_class)
            #计算每个epoch的精度
            score_dict_epoch = metric.get_metric_dict_per_epoch()
            self.epoch_mf1 = score_dict_epoch['mF1']
            message = 'Training: Epoch %d / %d\n' % (epoch_id, self.train_cfg['num_epoch']-1)
            message_class_list = ['{key}: {value:.5f} '.format(key=key, value=value.item()) for key, value in score_dict_epoch.items()]
            message_class = ''.join(map(str, message_class_list)) + '\n\n'
            print(message + message_class)
            #开始验证
            print("......Begin Evaluation......")
            metric.clear()
            self.mode.eval()
            m_val = len(val_dataloader)
            with torch.no_grad():
                for batch_id, data in enumerate(val_dataloader):
                    val_img, val_label = data
                    val_img = val_img.to(device)
                    val_label = val_label.to(device)

                    val_logits, val_prob = self.mode(val_img)
                    val_loss = criterion(val_logits, val_label)
                    if np.mod(batch_id, 20) == 1:
                        label = val_label.detach().squeeze()
                        pred = torch.argmax(val_prob.detach(), dim=1)
                        metric.update_confusion_matrix(gt=label, pred=pred)
                        score_dict_batch = metric.get_matrix_per_batch(gt=label, pred=pred)
                        message = 'Evaluation: [%d,%d][%d,%d], val_loss: %.5f\n' % (
                        epoch_id, self.train_cfg['num_epoch'] - 1, batch_id, m_val, val_loss.item())
                        message_class_list = ['{key}: {value:.5f} '.format(key=key, value=value.item()) for key, value in
                                              score_dict_batch.items()]
                        message_class = ''.join(map(str, message_class_list)) + '\n'
                        print(message + message_class)
                # 计算每个epoch的精度
                score_dict_epoch = metric.get_metric_dict_per_epoch()
                self.epoch_mf1 = score_dict_epoch['mF1']
                message = 'Evaluation: Epoch %d / %d\n' % (epoch_id, self.train_cfg['num_epoch'] - 1)
                message_class_list = ['{key}: {value:.5f} '.format(key=key, value=value.item()) for key, value in
                                      score_dict_epoch.items()]
                message_class = ''.join(map(str, message_class_list)) + '\n\n'
                print(message + message_class)
            #保存当前模型且判断最优模型
            save_checkpoint(ckpt_name='last_ckpt.pt', epoch_id=epoch_id,
                            model_dict=self.mode.state_dict(), path=checkpoint_path)
            if self.epoch_mf1 > self.best_val_mf1:
                self.best_val_mf1 = self.epoch_mf1
                self.best_epoch_id = epoch_id
                save_checkpoint(ckpt_name='best_ckpt.pt', epoch_id=self.best_epoch_id,
                                model_dict=self.mode.state_dict(), path=checkpoint_path)



if __name__=='__main__':
    seg_train = SegInterface()




