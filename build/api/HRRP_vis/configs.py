# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        config.py
# Author:           JunJie Ren
# Version:          v1.1
# Created:          2021/06/13
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 参数配置文件
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    None
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/13 |          creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v1.1    | 2020/07/09 |     增加ACARS配置部分
# ------------------------------------------------------------------------
'''

class DefaultConfigs_HRRP(object):
    ''' 默认参数配置 '''
    # Dataset
    trainTxtPath = './data/train.txt'
    validTxtPath = './data/valid.txt'
    dataset_name = "HRRP"                # RML2016.04c(6db,50): 89.0689
    num_classes = 5                            # 分类类别数
    signal_len = "4096, 2"
    process_IQ = True                           # 是否在载入数据时对IQ两路进行预处理

    batch_size = 1                            # DataLoader中batch大小，550/110=5 Iter
    num_workers = 4                             # DataLoader中的多线程数量

    # model
    model = "CNN_HRRP512"                   # 指定模型，MsmcNet_ACARS or MsmcNet_RML2016
    resume = False                            # 是否加载训练好的模型
    checkpoint_name = 'CNN_HRRP512.pth' # 训练完成的模型名

    # train
    num_epochs = 50                          # 训练轮数
    lr = 0.01                                   # 初始lr
    valid_freq = 1                              # 每几个epoch验证一次
    
    # log
    iter_smooth = 1                             # 打印&记录log的频率

    # seed = 1000                               # 固定随机种子

    # CAM
    Erase_thr = 0.3                             # CAM擦除软阈值，越小MASK保留越多，擦除越多
    CAM_omega = 20                              # 该参数从GAIN论文中获得，理论上将CAM的尺度拓展到0-omega，值越小越软


cfgs = DefaultConfigs_HRRP()
