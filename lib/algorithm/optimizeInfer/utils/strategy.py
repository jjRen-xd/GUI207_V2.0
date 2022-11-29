# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        utils/strategy.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 项目中用到的一些工具函数杂项，包括学习率调整、准确率计算
                    等，需不断更新
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    <0> step_lr():
                        -- 学习率手动调整，依据轮次调整，与keras代码中保持一致
                    <1> accuracy():
                        -- 计算top k分类准确率
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/14 | 修改学习率调整策略，修复bug
# ------------------------------------------------------------------------
'''
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def step_lr(epoch, lr):
    learning_rate = lr
    if epoch < 12:
        learning_rate = lr
    elif epoch % 12 == 0:
        learning_rate = lr * 0.5
    return learning_rate


def accuracy(output, target, topk=(1, 5)):
    """ 
    Funcs:
        计算top k分类准确率 
    Args:
        output: 一个batch的预测标签
        target：一个batch的真实标签
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # [110,1]
    # targ = torch.argmax(target, axis=1) #one-hot才需要加这一句
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  #用one-hot这里有个问题：target[bz,6],pred[1,bz]
     #RML:pred [110,1]，上句是使target从[110]变成[1,110]
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def drawConfusionMatrix(classes, savepath, output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True) # [110,1]
    pred = pred.t()
    confusionMatrix = confusion_matrix(target.cpu(), pred.reshape((-1)).cpu())

    plt.figure()
    proportion = []
    length = len(confusionMatrix)
    for i in confusionMatrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    config = {"font.family": 'Times New Roman'}  # 设置字体类型
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12, rotation=20)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = confusionMatrix.max() / 2.

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusionMatrix.size, 2))
    for i, j in iters:
        if i == j:
            plt.text(j, i + 0.12, format(confusionMatrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5)  # 显示对应的数字
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i + 0.12, format(confusionMatrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(savepath[:savepath.rfind('/')]+'/confusion_matrix_temp.jpg', dpi=300)
