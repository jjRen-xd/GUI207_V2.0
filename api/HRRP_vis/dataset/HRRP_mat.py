# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        dataset/HRRP_mat.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2022/08/18
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 数据集HRRP处理载入程序
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/confgs.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       <0> HRRPDataset(Dataset): 
                        -- 定义HRRPDataset类,继承Dataset方法,并重写
                        __getitem__()和__len__()方法
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/08/18 |   完成HRRP数据载入功能
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v1.1    | 2022/11/16 | 数据从.txt结构改为.mat结构
--------------------------------------------------------------------------
'''

import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class HRRPDataset(Dataset):
    ''' 定义HRRPDataset类,继承Dataset方法,并重写__getitem__()和__len__()方法 '''
    def __init__(self, datasetPath, transform=None):
        ''' 初始化函数,得到数据 '''
        self.signals, self.labels, _ = read_mat(datasetPath)
        print(_)
        self.transform = transform

    def __getitem__(self, index):
        ''' index是根据batchsize划分数据后得到的索引,最后将data和对应的labels进行一起返回 '''
        data = self.signals[index][None, :, None]
        label = self.labels[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        ''' 该函数返回数据大小长度,目的是DataLoader方便划分 '''
        return len(self.labels)


def read_mat(folder_path):
    """
        Func:
            从.mat文件读取数据并预处理
        Args:
            folder_path: 数据集路径
        Return:
            signals: 数据
            labels: 标签
            folder_name: 数据标签名
    """
    # 读取路径下所有文件夹的名称并保存
    file_name = os.listdir(folder_path)     # 读取所有文件夹,将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序

    signals = []
    labels = []
    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的.mat文件名称
        class_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[0]   # 类别的.mat文件路径
        matrix_base = os.path.basename(class_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        class_data = scio.loadmat(class_path)[matrix_name].T  # 读入.mat文件,并转置

        class_data_normalization = data_normalization(class_data)  # 归一化处理
        # 设置标签
        label = np.zeros((len(class_data_normalization), len(folder_name)))
        label[:, i] = 1

        signals += class_data_normalization.tolist()
        labels += label.tolist()
    return np.array(signals), np.array(labels), folder_name


def data_normalization(data):
    """
        Func:
            数据归一化
        Args:
            data: 待归一化的数据
        Return:
            data: 归一化后的数据
    """
    for i in range(0, len(data)):
        data[i] -= np.min(data[i])
        data[i] /= np.max(data[i])
    return data



if __name__ == "__main__":
    ''' 测试HRRP.py,测试dataLoader是否正常读取、处理数据 '''

    transform = transforms.Compose([ 
                                    # waiting add
                                    ])
    # 通过ACARSDataset将数据进行加载,返回Dataset对象,包含data和labels
    dataset = HRRPDataset(datasetPath='./dataset/HRRP_simulate_train_512xN_c5', transform=transform)
    # 通过DataLoader读取数据
    hrrpLoader = DataLoader( dataset, \
                        batch_size=160, \
                        num_workers=4, \
                        shuffle=True, \
                        drop_last=False)
    for data, i in tqdm(hrrpLoader):
        print("Size:", data.shape)