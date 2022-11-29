# -*- coding: utf-8 -*-    #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
from sklearn import preprocessing
def ff(data):
    data=data.T
    min_max_scaler = preprocessing.MinMaxScaler()
    data_norm = min_max_scaler.fit_transform(data)
    print(data_norm.shape)
    plt.figure() # figsize=(4,3)
    plt.imshow(data_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()#图紧凑
    plt.colorbar()#显示色带
    plt.xlim((0, 128))#x轴长度
    my_x_ticks = np.arange(0, 128, 16)#x轴刻度
    plt.xticks(my_x_ticks)#x轴刻度
    plt.yticks(alpha=0)#隐藏刻度数字
    plt.tick_params(axis='y', width=0)#隐藏刻度线
    plt.savefig('./colorMap.png', dpi=300)
    plt.close()
#x = np.full((128,50),99999, dtype = int)
#data = np.empty([128,64], dtype = int)

#ff(data)