'''
# ------------------------------------------------------------------------
# File Name:        hrrp.py
# Author:           Chang
# Version:          v1.0
# Created:          2022/10/24
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — —
#                           --> 小样本信号(RML2016.04c)系列代码 <--
#                   -- 对HPPR仿真数据集进行处理：.npz数据其实是将几个.npy压缩
#                   后得到的
#                   -- 保存格式：
#                   -- 数据集描述：6个类别，标签是one-hot形式，已经划分好了测试
#                   与训练集(5:5)
#                   --TODO: 1.one-hot标签转为序列 （是否必要？）
#                           2.进网络的大小要变成(bz,1,128,2)，现在为(1,128) DONE
#                           3.对每个样本进行归一化  DONE
#                   — — — — — — — — — — — — — — — — — — — — — — — — — — —
# Module called:    None
# Function List:    None
# Class List:       None
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# History:
#      |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  <0> | Chang |   v1.0    | 2022/10/24 |   Achieve all functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
import numpy as np
import torch, os
import scipy.io as sio

def loadNpz():
    path = './dataset/HRRP/hrrp-15_128.npz'

    dataset = np.load(path, allow_pickle=True)
    #print(dataset.files)  # ['train_X', 'train_Y', 'test_X', 'test_Y']
    #print(type(dataset['train_X']))  # 数组
    #print(dataset['test_X'].shape)  # (275,128)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for sam in dataset['train_X']:
        sam = np.around((sam - sam.min()) / (sam.max() - sam.min()), decimals=4)  # 归一化
        sample = (sam.reshape(1, sam.size)).repeat(2, axis=0).T
        x_train.append(sample)
    x_train = np.asarray(x_train)[:, np.newaxis, :, :]  # (275,1,128,2)
    for lab in dataset['train_Y']:
        lab = np.argmax(lab, axis=0)
        y_train.append(lab)
    y_train = np.asarray(y_train)  # (275,)
    #print(y_train.shape)
    for sam in dataset['test_X']:
        sam = np.around((sam - sam.min()) / (sam.max() - sam.min()), decimals=4)
        sample = (sam.reshape(1, sam.size)).repeat(2, axis=0).T
        x_test.append(sample)
    x_test = np.asarray(x_test)[:, np.newaxis, :, :]
    for lab in dataset['test_Y']:
        lab = np.argmax(lab, axis=0)
        y_test.append(lab)
    y_test = np.asarray(y_test)
    return x_train, y_train, x_test, y_test


# 数据归一化
def data_normalization(data):
    DATA = []
    for i in range(0, len(data)):
        data_max = max(data[i])
        data_min = min(data[i])
        data_norm = []
        for j in range(0, len(data[i])):
            data_one = np.around((data[i][j] - data_min) / (data_max - data_min), decimals=4)
            data_norm.append(data_one)
        DATA.append(data_norm)
    DATA = np.array(DATA)
    return DATA


# 从.mat文件读取数据并预处理
def loadAllMat(read_path):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序

    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的.mat文件名称
        class_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[0]   # 类别的.mat文件路径
        matrix_base = os.path.basename(class_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        class_data = sio.loadmat(class_path)[matrix_name].T  # 读入.mat文件，并转置

        class_data_normalization = data_normalization(class_data)  # 归一化处理
        #if(i==1):
            #print("class_data_normalization.shape==",class_data_normalization.shape)
            #print("class_data_normalization[50]==",class_data_normalization[50])
        # 数据复制2次
        class_data_picture = []
        for j in range(0, len(class_data_normalization)):
            class_data_one = class_data_normalization[j]
            empty = np.zeros((len(class_data_one), 2))
            #empty = np.zeros((2, len(class_data_one)))
            for k in range(0, len(class_data_one)):
                empty[k, :] = class_data_one[k]
            class_data_picture.append(empty)
        class_data_picture = np.array(class_data_picture)   # 列表转换为数组

        # 设置标签
        label = np.zeros(len(class_data_normalization),dtype='int64')
        label[:] = int(i)

        # 划分训练数据集和测试数据集
        x_train = class_data_picture[:int(len(class_data_picture)/2), :]
        x_test = class_data_picture[int(len(class_data_picture)/2):, :]
        y_train = label[:int(len(class_data_picture)/2)]
        y_test = label[int(len(class_data_picture)/2):]
        #print("x_trian.shape=",x_train.shape)
        #print("x_test.shape=",x_test.shape)
        if i == 0:
            train_x = x_train
            test_x = x_test
            train_y = y_train
            test_y = y_test
        else:
            train_X = np.concatenate((train_x, x_train))
            train_Y = np.concatenate((train_y, y_train))
            test_X = np.concatenate((test_x, x_test))
            test_Y = np.concatenate((test_y, y_test))

            train_x = train_X
            train_y = train_Y
            test_x = test_X
            test_y = test_Y
    # args.len = len(train_X)
    train_X=np.expand_dims(train_X,axis=1)
    test_X=np.expand_dims(test_X,axis=1)
    return train_X, train_Y, test_X, test_Y

def loadSample(class_path,sampleIndex):
    
    
    matrix_base = os.path.basename(class_path)
    matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
    class_data = sio.loadmat(class_path)[matrix_name].T  # 读入.mat文件，并转置
    class_data_normalization = data_normalization(class_data)  # 归一化处理

    matColNum=len(class_data_normalization)
    sampleIndex = int(sampleIndex%(matColNum/2) + matColNum/2)

    # 数据复制2次
    class_data_picture = []
    for j in range(0, matColNum):
        class_data_one = class_data_normalization[j]
        empty = np.zeros((len(class_data_one), 2))
        #empty = np.zeros((2, len(class_data_one)))
        for k in range(0, len(class_data_one)):
            empty[k, :] = class_data_one[k]
        class_data_picture.append(empty)
    class_data_picture = np.array(class_data_picture)   # 列表转换为数组 [100,128,2]
    ret = np.expand_dims(class_data_picture[sampleIndex],axis=0) # [1,128,2]
    ret = np.expand_dims(ret,axis=0)                            # [1,1,128,2]
    return ret 