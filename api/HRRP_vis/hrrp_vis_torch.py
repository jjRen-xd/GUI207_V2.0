# -*- coding: utf-8 -*- #

import os
import scipy.io as scio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import gc
import shutil

import argparse

from utils.CAM import t2n, HookValues, GradCAM, GradCAMpp, XGradCAM, EigenGradCAM, LayerCAM


name2label = {
    "DQ": 0,
    "DT": 1,
    "QDZ": 2,
    "XQ": 3,
    "Z": 4
}

def vis_fea(checkpoint_path, vis_layer, signal, label):
    # model
    model = torch.load(checkpoint_path)    # load checkpoint
    # print(model)
    model.cuda()

    try:
        target_layer = eval(f'model.{vis_layer}')
    except Exception as e:
        print(model)
        raise RuntimeError('layer does not exist', e)
    hookValues = HookValues(target_layer)
    signal = torch.from_numpy(signal).cuda().float()

    # forward
    logits = model(signal)
    logits = torch.sigmoid(logits)

    # bakward
    batch_size, _ = logits.shape
    _range = torch.arange(batch_size)
    pred_scores = logits[_range, label]
    pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)

    # Capture activations and gradients
    activations = hookValues.activations       # ([1, 15, 124, 1])
    gradients = hookValues.gradients           # ([1, 15, 124, 1])

    return t2n(activations), t2n(gradients)
        

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


def read_mat(matPath):
    ''' 读取.mat文件 '''
    # mat = scio.loadmat(matPath)
    matrix_base = os.path.basename(matPath)
    labelName = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
    signals = scio.loadmat(matPath)[labelName].T  # 读入.mat文件,并转置
    signals_normalization = data_normalization(signals)  # 归一化处理
    labels = [name2label[labelName]] * len(signals_normalization)  # 标签   

    return signals_normalization, labels, labelName


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MMDet Visualize a model'
    )
    parser.add_argument(
        '--checkpoint', 
        default="./checkpoints/CNN_HRRP512.pth",
        type=str,
        help='checkpoint file'
    )
    parser.add_argument(
        '--visualize_layer',
        default="Block_3[0]",
        type=str,
        help='Name of the hidden layer of the model to visualize'
    )
    parser.add_argument(
        '--mat_path',
        default="./dataset/HRRP_simulate_test_512xN_c5/DQ/DQ.mat",
        type=str,
        help='The .mat path of signal to visualize'
    )
    parser.add_argument(
        '--mat_idx',
        default=0,
        type=int,
        help='The .mat index of visual signal'
    )
    parser.add_argument(
        '--act_or_grad',
        default=0,
        type=int,
        help='Visual features or gradients'
    )
    parser.add_argument(
        '--save_path',
        default="./figs/fea_output",
        type=str,
        help='The path of feature map to save'
    )
    args = parser.parse_args()

    # 读取数据
    signals, labels, labelName = read_mat(args.mat_path)
    signal = signals[args.mat_idx][None, None, :, None]
    label = labels[args.mat_idx]  

    # 捕获特征
    activations, gradients = vis_fea(args.checkpoint, args.visualize_layer, signal, label)    # bz, nc, sig_len, 1

    # 检查保存路径
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # 保存激活图/梯度图
    if args.act_or_grad == 0:
        visFeatures = activations[0]
    elif args.act_or_grad == 1:
        visFeatures = gradients[0]
        # print(np.max(visFeatures), np.min(visFeatures))
        if np.max(np.abs(visFeatures)) != 0:    # 规整到一个合适的范围
            visFeatures *= 1/np.max(np.abs(visFeatures))
    else:
        raise RuntimeError('args.act_or_grad must be 0 or 1')

    if visFeatures.ndim == 1:
        plt.figure(figsize=(18, 4), dpi=100)
        plt.grid(axis="y")
        plt.bar(range(len(visFeatures)), visFeatures)
        plt.title("Signal Type: "+labelName+"    Model Layer: "+"model."+args.visualize_layer)
        plt.xlabel("Number of units")
        plt.ylabel("Activation value")
        plt.savefig(args.save_path + "/FC_vis.png")

        plt.clf()
        plt.close()
        gc.collect()
    else:
        for idx, featureMap in enumerate(visFeatures):   # for every channels
            plt.figure(figsize=(18, 4))
            plt.title("Signal Type: "+labelName+"    Model Layer: "+"model."+args.visualize_layer)
            plt.xlabel('N')
            plt.ylabel("Value")
            plt.plot(featureMap[:, 0], linewidth=2, label = 'Hidden layer features')
            plt.legend(loc="upper right")
            plt.savefig(args.save_path + "/" + str(idx+1) + ".png")
            # plt.show()

            plt.clf()
            plt.close()
            gc.collect()

    print("finished")


