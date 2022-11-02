# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        dataset/RML2016.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/07/09
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 数据集ACARS处理载入程序
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/confgs.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> loadNpy_acars(): 
                        -- 从*.npy中载入原始训练、测试数据,并进行IQ预处理、维
                        度调整,对训练数据进行随机打乱
                    <1> processIQ():
                        -- 对两路信号分别进行预处理，结合两路为复数，除以标准
                        差，再分离实部虚部到两路
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       <0> RMLDataset(Dataset): 
                        -- 定义ACARSDataset类，继承Dataset方法，并重写
                        __getitem__()和__len__()方法
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/07/09 |   完成ACARS数据载入功能
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> |
--------------------------------------------------------------------------
'''

import sys
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

sys.path.append("../")
from configs import cfgs
from dataset.RML2016_10a.classes import modName


class ACARSDataset(Dataset):
    ''' 定义ACARSDataset类，继承Dataset方法，并重写__getitem__()和__len__()方法 '''
    def __init__(self, data_root, data_label, transform=None):
        ''' 初始化函数，得到数据 '''
        self.data = data_root
        self.label = data_label
        self.transform = transform

    def __getitem__(self, index):
        ''' index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回 '''
        data = self.data[index]
        labels = self.label[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, labels

    def __len__(self):
        ''' 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼 '''
        return len(self.data)


def loadNpy_acars(train_path_x, train_path_y, test_path_x, test_path_y, process_IQ=False):
    """
    Funcs:
        从*.npy中载入原始训练、测试数据,并进行IQ预处理、维度调整,对训练数据进行随机打乱
    Args:
        train_path: str
            example: "x_train_50persample.npy"衍生"y_train_50persample.npy"
        test_path: str
            example: 
        process_IQ: Bool
    Returns:
        original train data:  
            x_train: numpy.ndarray(dtype=np.float64, shape(SampleNum, 1, 4096, 2))
                example: (2000, 1, 4096, 2)
            y_train: numpy.ndarray(dtype=np.int64, shape(SampleNum))
                example: (2000)
        original test data: 
            x_test: numpy.ndarray(dtype=np.float64, shape(SampleNum, 1, 4096, 2))
                example: (5500, 1, 4096, 2)
            y_test: numpy.ndarray(dtype=np.int64, shape(SampleNum))
                example: (5500)
    """

    x_train = np.load(train_path_x)         # (4096, 2, 1, 2000)
    y_train = np.load(train_path_y)         # (2000)
    x_test = np.load(test_path_x)
    y_test = np.load(test_path_y) 

    # PyTorch为(N, C, H, W), 需更改维度  
    x_train = np.swapaxes(x_train, 0, 3)    # (2000, 2, 1, 4096)
    x_train = np.swapaxes(x_train, 1, 2)    # (2000, 1, 2, 4096)
    x_train = np.swapaxes(x_train, 2, 3)    # (2000, 1, 4096, 2)

    x_test = np.swapaxes(x_test, 0, 3)    # (2000, 2, 1, 4096)
    x_test = np.swapaxes(x_test, 1, 2)    # (2000, 1, 2, 4096)
    x_test = np.swapaxes(x_test, 2, 3)    # (2000, 1, 4096, 2)

    # 训练数据随机打乱
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    x_train = x_train[index,:,:,:]
    y_train = y_train[index]
    
    # IQ两路预处理
    if process_IQ:
        x_train, x_test = processIQ(x_train, x_test)
    return x_train, y_train, x_test, y_test


def processIQ(x_train, x_test):
    ''' 对两路信号分别进行预处理，结合两路为复数，除以标准差，再分离实部虚部到两路 '''
    for sample in x_train:
        sample_complex = sample[0, :, 0] + sample[0, :, 1] * 1j
        sample_complex = sample_complex / np.std(sample_complex)
        sample[0, :, 0] = sample_complex.real
        sample[0, :, 1] = sample_complex.imag
    for sample in x_test:
        sample_complex = sample[0, :, 0] + sample[0, :, 1] * 1j
        sample_complex = sample_complex / np.std(sample_complex)
        sample[0, :, 0] = sample_complex.real
        sample[0, :, 1] = sample_complex.imag
    return x_train, x_test


if __name__ == "__main__":
    ''' 测试ACARS.py，测试dataLoader是否正常读取、处理数据 '''
    # 读取原始numpy
    x_train, y_train, x_test, y_test = loadNpy_acars(
        cfgs.train_path_x, 
        cfgs.train_path_y, 
        cfgs.test_path_x, 
        cfgs.test_path_y, 
        cfgs.process_IQ
    )
    
    transform = transforms.Compose([ transforms.ToTensor()
                                    # waiting add
                                    ])
    # 通过ACARSDataset将数据进行加载，返回Dataset对象，包含data和labels
    torch_data = ACARSDataset(x_train, y_train, transform=transform)
    # 通过DataLoader读取数据
    datas = DataLoader( torch_data, \
                        batch_size=cfgs.batch_size, \
                        num_workers=cfgs.num_workers, \
                        shuffle=True, \
                        drop_last=False)
    for i, data in enumerate(datas):
        # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
        print("第 {} 个Batch \n{}".format(i, data))
        print("Size：", len(data[0]))
    
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)