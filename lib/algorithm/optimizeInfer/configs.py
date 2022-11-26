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



class DefaultConfigs_RML(object):
    ''' 默认参数配置 '''
    # Dataset
    dataset_name = "RML2016.04c"  # RML2016.04c(6db,50): 89.0689
    num_classes = 11  # 分类类别数
    signal_len = "128,2"
    train_path = '/media/hp3090/gc/Signal_master/04c-Generated-data/6dB_SNR/6dB-SNR_7288-train.npy'
    # 原始训练数据目录，D:\WIN_10_DEDKTOP\onWorking\SAR&DD\PyTorch_Codes\Signal_master\dataset\RML2016_04c\\6dB-SNR_50-samples.npy
    test_path = '/media/hp3090/gc/Signal_master/04c-Generated-data/6dB_SNR/6dB-SNR_815-test.npy'
    # 原始测试数据目录，D:\WIN_10_DEDKTOP\onWorking\SAR&DD\PyTorch_Codes\Signal_master\dataset\RML2016_04c\\6dB-SNR_4055-test.npy
    process_IQ = True  # 是否在载入数据时对IQ两路进行预处理

    batch_size = 1000  # DataLoader中batch大小，550/110=5 Iter,原本是110
    num_workers = 4  # DataLoader中的多线程数量

    # model
    model = "MsmcNet_RML2016"  # 指定模型，目前就一个
    resume = False  # 是否加载训练好的模型  TODO:
    checkpoint_name2 = 'MsmcNet_RML2016_04c_4db_use5.pth'  # 训练完成的模型名 TODO:
    checkpoint_name1 = 'MsmcNet_RML2016_04c_6db_original.pth'

    # train
    num_epochs = 300  # 训练轮数
    lr = 0.01  # 初始lr=0.01  TODO:
    valid_freq = 10  # 每几个epoch验证一次  TODO:

    # log
    iter_smooth = 5  # 打印&记录log的频率

    # seed = 1000                               # 固定随机种子
    # CAM，原本没有这两行
    Erase_thr = 0.3  # CAM擦除软阈值，越小MASK保留越多，擦除越多
    CAM_omega = 20  # 该参数从GAIN论文中获得，理论上将CAM的尺度拓展到0-omega，值越小越软


class DefaultConfigs_HRRP(object):
    ''' 默认参数配置 '''
    # Dataset
    dataset_name = "hrrp"  #
    num_classes = 6  # 分类类别数
    signal_len = "128,2"

    batch_size = 1000  # DataLoader中batch大小，550/110=5 Iter,原本是110
    num_workers = 4  # DataLoader中的多线程数量

    # model
    model = "MsmcNet_HRRP"  # 指定模型，目前就一个
    resume = False  # 是否加载训练好的模型  TODO:
    checkpoint_name2 = 'MsmcNet_RML2016_04c_4db_use5.pth'  # 训练完成的模型名 TODO:
    checkpoint_name1 = 'MsmcNet_hrrp_original.pth'

    # train
    num_epochs = 50  # 训练轮数
    lr = 0.01  # 初始lr=0.01  TODO:
    valid_freq = 2  # 每几个epoch验证一次  TODO:

    # log
    iter_smooth = 5  # 打印&记录log的频率

    # seed = 1000                               # 固定随机种子
    # CAM，原本没有这两行
    Erase_thr = 0.3  # CAM擦除软阈值，越小MASK保留越多，擦除越多
    CAM_omega = 20  # 该参数从GAIN论文中获得，理论上将CAM的尺度拓展到0-omega，值越小越软


cfgs = DefaultConfigs_HRRP()
# cfgs = DefaultConfigs_RML()
