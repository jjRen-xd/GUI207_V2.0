# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        networks/MsmcNet.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — —
                            --> DD信号识别（可解释）系列代码 <--
                    -- MsmcNet网络结构定义
                    — — — — — — — — — — — — — — — — — — — — — — — — — — —
# Module called:    None
# Function List:    None
                    — — — — — — — — — — — — — — — — — — — — — — — — — — —
# Class List:       <0> MsmcNet_RML2016():
                        -- MsmcNet网络结构定义，所有参数均按照MsmcNet的keras
                        版本设置
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   <0> | JunJie Ren |   v1.0    | 2020/06/14 |    实现MsmcNet网络的定义
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   <0> | JunJie Ren |   v1.1    | 2020/07/09 | 增加 for ACARS网络的定义
# ------------------------------------------------------------------------
'''

import sys
import torch
import torch.nn as nn
# from torchsummary import summary


class MsmcNet_RML2016(nn.Module):
    '''
    Funcs:
        RML2016数据集下的MsmcNet网络结构定义，所有参数均按照MsmcNet的keras版本设置
    Network input size:
        convention: (N, 1, 128, 2)
    Notes：
        <1> PyTorch中conv2d无padding='same'，经手动调整padding大小保证一致
        <2> FIXME PyTorch在网络定义中无法加入正则项：kernel_regularizer=l2(0.001)
        <3> FIXME PyTorch中的BN的参数量与kears不一致，仅为其一半
    '''
    def __init__(self, num_classes):
        super(MsmcNet_RML2016, self).__init__()

        self.Block_1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(5, 2), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
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
        self.fc1 = nn.Linear(75, 100)  #128：in_75
        self.fc0 = nn.Linear(375,100)  #225/375/975
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x, cams=None,S_masks=None,B_masks=None):  #x最开始[110,1,128,2]
        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_3(x)  #[110,15,14,1]
        x = self.Block_4(x)  #[110,15,5,1]  256:[bz,15,13,1]
        if cams is not None:
            S_masks_m = self.maxpool(S_masks)
            B_masks_m = self.maxpool(B_masks)
            S_masks_a = self.avgpool(S_masks)
            B_masks_a = self.avgpool(B_masks)
            Sa_x_m = x*S_masks_m  #f(x,cams)  # TODO 计算得到前背景特征图，在这里一定要把np2tensor了
            Ba_x_m = x*B_masks_m  #g(x,cams)   # x:float32 S_masks/sa_x: float64
            Sa_x_a = x*S_masks_a
            Ba_x_a = x*B_masks_a
            x = torch.cat((x,Sa_x_m,Ba_x_m,Sa_x_a, Ba_x_a), dim=1)  #注意需要将requires_grad=True #增减 ,Sa_x_a, Ba_x_a
            x = self.flatten(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = self.flatten(x) # 128:[bz,75] 256:[bz,195]
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)

        return x


class MsmcNet_HRRP256(nn.Module):
    '''
    Funcs:
        RML2016数据集下的MsmcNet网络结构定义，所有参数均按照MsmcNet的keras版本设置
    Network input size:
        convention: (N, 1, 128, 2)
    Notes：
        <1> PyTorch中conv2d无padding='same'，经手动调整padding大小保证一致
        <2> FIXME PyTorch在网络定义中无法加入正则项：kernel_regularizer=l2(0.001)
        <3> FIXME PyTorch中的BN的参数量与kears不一致，仅为其一半
    '''
    def __init__(self, num_classes):
        super(MsmcNet_HRRP256, self).__init__()

        self.Block_1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(5, 2), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
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
        self.fc1 = nn.Linear(195, 100)  #128：in_75
        self.fc0 = nn.Linear(975,100)  #225/375/975
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x, cams=None,S_masks=None,B_masks=None):  #x最开始[110,1,128,2]
        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_3(x)
        x = self.Block_4(x)  # 256:[bz,15,13,1]
        if cams is not None:
            S_masks_m = self.maxpool(S_masks)
            B_masks_m = self.maxpool(B_masks)
            S_masks_a = self.avgpool(S_masks)
            B_masks_a = self.avgpool(B_masks)
            Sa_x_m = x*S_masks_m  #f(x,cams)  # TODO 计算得到前背景特征图，在这里一定要把np2tensor了
            Ba_x_m = x*B_masks_m  #g(x,cams)   # x:float32 S_masks/sa_x: float64
            Sa_x_a = x*S_masks_a
            Ba_x_a = x*B_masks_a
            x = torch.concat((x,Sa_x_m,Ba_x_m,Sa_x_a, Ba_x_a), dim=1)  #注意需要将requires_grad=True #增减   Sa_x_a, Ba_x_a
            x = self.flatten(x)
            x = self.fc0(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = self.flatten(x) #  256:[bz,195]
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)

        return x


class MsmcNet_ACARS(nn.Module):
    ''' 
    Funcs:
        ACARS数据集下的MsmcNet网络结构定义，所有参数均按照MsmcNet的keras版本设置
    Network input size:
        convention: (N, 1, 4096, 2)
    Notes：
        <1> PyTorch中conv2d无padding='same'，经手动调整padding大小保证一致
        <2> FIXME PyTorch在网络定义中无法加入正则项：kernel_regularizer=l2(0.001)
        <3> FIXME PyTorch中的BN的参数量与kears不一致，仅为其一半
    '''
    def __init__(self, num_classes):
        super(MsmcNet_ACARS, self).__init__()

        self.Block_1 = nn.Sequential(
            nn.Conv2d(1, 150, kernel_size=(15, 2), stride=(1, 2), padding=(7, 1)),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.Block_2 = nn.Sequential(
            nn.Conv2d(150, 150, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_3 = nn.Sequential(
            nn.Conv2d(150, 150, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_4 = nn.Sequential(
            nn.Conv2d(150, 150, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_5 = nn.Sequential(
            nn.Conv2d(150, 150, kernel_size=(3, 1), stride=1, padding=0),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_6 = nn.Sequential(
            nn.Conv2d(150, 150, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.Block_7 = nn.Sequential(
            nn.Conv2d(150, 150, kernel_size=(5, 1), stride=1, padding=0),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4200, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_3(x)
        x = self.Block_4(x)
        x = self.Block_5(x)
        x = self.Block_6(x)
        x = self.Block_7(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    ''' 测试MsmcNet.py，测试网络结构构建是否构建正确，并打印每层参数 '''
    model = MsmcNet_RML2016(num_classes=11)
    #model = MsmcNet_ACARS(num_classes=20)
    model.cuda()
    print(model)
    # 统计网络参数及输出大小
    #summary(model, (1, 4096, 2), batch_size=110, device="cuda")
    summary(model, (1, 128, 2), batch_size=110, device="cuda")