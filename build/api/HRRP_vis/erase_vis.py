# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/erase_vis.py
# Author:           JunJie Ren
# Version:          v1.1
# Created:          2021/06/15
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 擦除信号部分特征，观察准确率变换，最基础的可解释
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
        |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <0> | JunJie Ren |   v1.0    | 2020/08/17 |           creat
--------------------------------------------------------------------------
'''

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from configs import cfgs
from dataset.RML2016 import RMLDataset, loadNpy
from dataset.RML2016_10a.classes import modName
from dataset.ACARS import ACARSDataset, loadNpy_acars

from utils.strategy import accuracy
from utils.plot import plot_confusion_matrix
from utils.CAM import compute_gradcampp, compute_Sigcam
from utils.signal_vis import showImgSignal,showOriSignal,showCamSignal
from utils.Guided_BP import GuidedBackprop


def showCAM(cam_thr, reserve_erase):
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
    idx = 0
    change_great = 0

    for images, labels in dataloader_valid:
        images = Variable(images).cuda().float()
        labels = Variable(labels).cuda()
        for image, label in zip(images, labels):
            idx += 1
            c, w, h = image.shape
            ERASE_RES = 10                                    # 遮挡间隔分辨率
            score_change = []
            change_value = False

            # showOriSignal(t2n(image), label, idx)           # 绘制原始信号图
            softmax = nn.Softmax(dim=0)
            score_ori = t2n(softmax(model(image.unsqueeze(0))[0])[label])   # 原始得分

            '''擦除信号部分信息，重新计算得分变化'''
            # TODO
            # print(image.shape)  # torch.Size([1, 128, 2])
            # for iq in range(h):                  # 2
            for pos in range(w//ERASE_RES + 1):    # 128//10 = 12
                pos *= ERASE_RES

                masked_image = image.clone()
                if pos+ERASE_RES > w:
                    masked_image[0, pos:w, :] = 0
                else:
                    masked_image[0, pos:pos+ERASE_RES, :] = 0
                # showOriSignal(t2n(masked_image), label, idx)  # 绘制擦除后信号图
                score_erase = t2n(softmax(model(masked_image.unsqueeze(0))[0])[label])   # 原始得分
                if (score_ori-score_erase)>0.2:
                    change_value = True
                score_change.append(score_ori - score_erase)
            
            if change_value:
                change_great += 1
            # print(score_change)
            score_change = np.array(score_change)
            score_change = cv2.resize(score_change, (h, w), interpolation=cv2.INTER_NEAREST)
            showCamSignal(t2n(image), score_change, label, idx)
    print(change_great)

def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)


if __name__ == '__main__':
    showCAM(cam_thr = 0.3, reserve_erase = 0)
