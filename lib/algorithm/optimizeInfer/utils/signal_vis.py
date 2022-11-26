# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/utils/signal_vis.py
# Author:           JunJie Ren
# Version:          v1.1
# Created:          2021/06/15
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 可视化信号输入
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/config.py
                    <1> PATH_TOOT/dataset/RML2016.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> drawAllOriSignal():
                        绘制所有信号输入样本的图像，并保存至相应标签的文件夹下
                    <1> showOriSignal():
                        绘制并展示一个样本信号的图像
                    <2> showImgSignal():
                        绘制并展示一个信号样本的二维可视化图像
                    <3> showCamSignal():
                        叠加信号与CAM图，可视化CAM解释结果，并按类型保存
                    <4> mask_image():
                        软阈值擦除CAM对应的判别性特征区域 
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/15 |           creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v1.1    | 2020/07/09 | 优化无name的数据集调用问题
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   <2> | JunJie Ren |   v1.2    | 2020/07/13 |     增加CAM阈值擦除函数
--------------------------------------------------------------------------
'''

import sys
import os

import cv2
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg') #TkAgg

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   #解决路径报错
sys.path.insert(0,parentdir)  

sys.path.append("../")
from configs import cfgs
# from dataset.RML2016 import loadNpy
from skimage import measure
from matplotlib.path import Path
from numpy import * 

def t2n(t):
    return t.detach().cpu().numpy().astype(np.int)  #该端代码表示将tensor类型的变量t转换为numpy浮点型

def ave_mask_cam(signal,cam,label,idx):
    #进行cam二值化，输出包含最大响应值的外接矩阵四个顶点
    cam_max_arg = (np.squeeze(np.where(cam == np.max(cam)))).T  # DONE: 2*2 [52,52],[0,1],应该表示成[52,0],[52,1]
    ave_mask = np.where(cam <= np.mean(cam), 0, 1)
    labels = measure.label(ave_mask,connectivity=2)
    regions = measure.regionprops(labels)
    valid_label = set()
    global rectbox
    for region in regions:
        box = region.bbox   # (32,0,102,0)
        region_box = np.array([[box[0],box[1]],[box[0],box[3]],[box[2],box[1]],[box[2],box[3]]])
        p = Path([(region_box[0]),(region_box[1]),(region_box[2]),(region_box[3])])
        if p.contains_point(cam_max_arg[1]):  #出现在边框上的情况
            # valid_label.add(region.label) 画图再用
            rect = cv2.minAreaRect(region.coords[:, ::-1])
            rectbox = cv2.boxPoints(rect)  #float32
            break
    rectbox = np.array(rectbox,dtype = np.int32)  #[1,1,4,2]
    # rectbox = np.expand_dims(np.array(rectbox, dtype = np.int32),0) #从[4,2]到[1,4,2]fillPoly才能用 TODO: 一开始用的是这行
    Sa_mask = np.zeros_like(cam)   #Sa为显著区域，Ba是背景区域
    Ba_mask = np.ones_like(cam)
    # cv2.polylines(Sa_mask,rectbox,1,255)
    Sa_mask = cv2.fillPoly(Sa_mask, [rectbox], 1)  #rectbox要求是int32/64,Sa_mask是float64,1.类型转换 2.二维升三维 #[2,128]
    Ba_mask = cv2.fillPoly(Ba_mask,[rectbox],0)
    # Sa_masked = cv2.bitwise_and(cam,Sa_mask)
    # Ba_masked = cv2.bitwise_and(cam,Ba_mask)

    return Sa_mask,Ba_mask


# if __name__ == "__main__":
#     x_train, y_train, x_test, y_test = loadNpy(cfgs.train_path, cfgs.test_path)
#     print(x_train.shape, y_train.shape)
#     # drawAllOriSignal(X=x_train, Y=y_train)
#     for idx in range(len(x_train)):
#         showOriSignal(x_train[idx], y_train[idx],idx)
#         showImgSignal(x_train[idx], y_train[idx])
#         # showOriSignal(x_train[idx], y_train[idx])

