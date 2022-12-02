# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/inference.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 测试模型分类准确率，并绘制混淆矩阵
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/configs.py
                    <1> PATH_ROOT/dataset/RML2016.py
                    <3> PATH_ROOT/utils/strategy.py;plot.py
                    <4> PATH_ROOT/dataset/ACARS.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> inference():
                        -- 使用训练好的模型对测试集进行推理，测试分类准确率，
                        绘制混淆矩阵
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
        |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <0> | JunJie Ren |   v1.0    | 2020/06/14 |           creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <1> | JunJie Ren |   v1.1    | 2020/07/09 |    新增ACARS测试程序选项
--------------------------------------------------------------------------
'''

import os

import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from configs import cfgs
from dataset.RML2016 import RMLDataset, loadNpy
from dataset.ACARS import ACARSDataset, loadNpy_acars
from utils.strategy import accuracy
from utils.plot import plot_confusion_matrix
from dataset.RML2016_10a.classes import modName


def inference():
    # model
    model = torch.load(os.path.join('./checkpoints', cfgs.checkpoint_name))    # load checkpoint
    print(model)
    # model = torch.nn.DataParallel(model)  # 多卡预留
    model.cuda()
    
   # Dataset
    if cfgs.dataset_name == "RML2016.04c":
        x_train, y_train, x_test, y_test = loadNpy(
            cfgs.train_path,
            cfgs.test_path,
            cfgs.process_IQ
        )
        Dataset = RMLDataset
    elif cfgs.dataset_name == "ACARS":
        x_train, y_train, x_test, y_test = loadNpy_acars(
            cfgs.train_path_x,
            cfgs.train_path_y,
            cfgs.test_path_x,
            cfgs.test_path_y,
            cfgs.process_IQ
        )
        Dataset = ACARSDataset
    else :
        print('ERROR: No Dataset {}!!!'.format(cfgs.model))
        
    # Valid data
    # BUG,BUG,BUG,FIXME
    transform = transforms.Compose([ 
                                        # transforms.ToTensor()
                                        # waiting add
                                    ])

    valid_dataset = Dataset(x_test, y_test, transform=transform)
    dataloader_valid = DataLoader(valid_dataset, \
                                batch_size=cfgs.batch_size, \
                                num_workers=cfgs.num_workers, \
                                shuffle=True, \
                                drop_last=False)

    sum = 0
    val_top1_sum = 0
    labels = []
    preds = []
    model.eval()
    for ims, label in dataloader_valid:
        labels += label.numpy().tolist()

        input = Variable(ims).cuda().float()
        target = Variable(label).cuda()
        output = model(input)

        _, pred = output.topk(1, 1, True, True)
        preds += pred.t().cpu().numpy().tolist()[0]

        top1_val = accuracy(output.data, target.data, topk=(1,))
        
        sum += 1
        val_top1_sum += top1_val[0]
    avg_top1 = val_top1_sum / sum
    print('acc: {}'.format(avg_top1.data))
    if cfgs.dataset_name == "RML2016.04c":
        labels_ = modName
    elif cfgs.dataset_name == "ACARS":
        labels_ = range(cfgs.num_classes)
    plot_confusion_matrix(labels, preds, labels_)


if __name__ == '__main__':
    inference()
