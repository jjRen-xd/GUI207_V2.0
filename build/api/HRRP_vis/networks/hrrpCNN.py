# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        networks/MsmcNet.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2022/08/18
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 简单的适用于HRRP的CNN网络结构定义
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    None
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       <0> CNN_HRRP512():
                        -- CNN_HRRP512网络结构定义
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/08/18 |  实现CNN_HRRP512网络的定义
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> 
# ------------------------------------------------------------------------
'''

import sys
import torch
import torch.nn as nn
from torchinfo import summary


class CNN_HRRP512(nn.Module):
    ''' 
    Funcs:
       初步为可视化HRRP所搭建的CNN_HRRP512网络结构
    Network input size:
        convention: (N, 1, 512, 1)
    Notes:
        <1> TODO
    '''
    def __init__(self, num_classes):
        super(CNN_HRRP512, self).__init__()

        self.Block_1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_2 = nn.Sequential(
            nn.Conv2d(30, 25, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_3 = nn.Sequential(
            nn.Conv2d(25, 15, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_4 = nn.Sequential(
            nn.Conv2d(15, 15, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(435, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_3(x)
        x = self.Block_4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    ''' 测试hrrpCNN.py,测试网络结构构建是否构建正确,并打印每层参数 '''
    model = CNN_HRRP512(num_classes=6)
    model.cuda()
    print(model)
    # 统计网络参数及输出大小
    summary(model, (110, 1, 512, 1), device="cuda")
