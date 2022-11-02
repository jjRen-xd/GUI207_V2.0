# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        dataset/HRRP.py
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
                        -- 定义HRRPDataset类，继承Dataset方法，并重写
                        __getitem__()和__len__()方法
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/08/18 |   完成HRRP数据载入功能
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> |
--------------------------------------------------------------------------
'''

from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class HRRPDataset(Dataset):
    ''' 定义HRRPDataset类，继承Dataset方法，并重写__getitem__()和__len__()方法 '''
    def __init__(self, idxTxtPath, transform=None):
        ''' 初始化函数，得到数据 '''
        self.dataPaths, self.labels = read_idx_txt(idxTxtPath)
        self.transform = transform

    def __getitem__(self, index):
        ''' index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回 '''
        data = loadData_FromTXT(self.dataPaths[index])
        label = self.labels[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        ''' 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼 '''
        return len(self.dataPaths)


def loadData_FromTXT(txtPath):
    ''' 从txt文件中读取数据, (512, ) '''
    data = np.loadtxt(txtPath)[:, 1]
    data -= np.min(data)
    data /= np.max(data)

    data = data[np.newaxis, :, np.newaxis]

    return data


def read_idx_txt(path):
    data, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(' ')
            data.append(im)
            labels.append(int(label))
    return data, labels


if __name__ == "__main__":
    ''' 测试HRRP.py，测试dataLoader是否正常读取、处理数据 '''

    transform = transforms.Compose([ 
                                    # waiting add
                                    ])
    # 通过ACARSDataset将数据进行加载，返回Dataset对象，包含data和labels
    dataset = HRRPDataset(idxTxtPath='./data/train.txt', transform=transform)
    # 通过DataLoader读取数据
    hrrpLoader = DataLoader( dataset, \
                        batch_size=160, \
                        num_workers=4, \
                        shuffle=True, \
                        drop_last=False)
    for data, i in tqdm(hrrpLoader):
        # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
        # print("第 {} 个Batch \n{}".format(i, data))
        # print("Size:", len(data[0]))
        print("Size:", data.shape)