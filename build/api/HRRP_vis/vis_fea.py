# -*- coding: utf-8 -*- #

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
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

def vis_fea(checkpoint_path, vis_layer, signal, name):
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

    # Calculate CAM
    activations = hookValues.activations       # ([1, 15, 124, 1])

    return t2n(activations)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MMDet Visualize a model'
    )
    parser.add_argument(
        '--checkpoint', 
        default="/media/z840/HDD_1/LINUX/DD_vis/HRRP_vis/checkpoints/CNN_HRRP512.pth",
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
        '--signal_path',
        default="/media/z840/HDD_1/LINUX/DD_vis/HRRP_vis/dataset/HRRP_20220508/DQ/0-0.5.txt",
        type=str,
        help='The path of image to visualize'
    )
    parser.add_argument(
        '--save_path',
        default="/media/z840/HDD_1/LINUX/DD_vis/HRRP_vis/figs/fea_output",
        type=str,
        help='The path of feature map to save'
    )
    args = parser.parse_args()


    # 读取数据
    signals = np.loadtxt(args.signal_path)[:, 1]
    signals -= np.min(signals)
    signals /= np.max(signals)
    signals = signals[None, None, :, None]
    CALSSES = [args.signal_path.split("/")[-2]]

    # 捕获特征
    activations = vis_fea(args.checkpoint, args.visualize_layer, signals, CALSSES)    # bz, nc, sig_len, 1

    # 保存图像
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    for idx, featureMap in enumerate(activations[0]):   # for every channels
        plt.figure(figsize=(18, 4))
        plt.title("Signal Type: "+CALSSES[0]+"    Model Layer: "+"model."+args.visualize_layer)
        plt.xlabel('N')
        plt.ylabel("Value")
        plt.plot(featureMap[:, 0], linewidth=2, label = 'Hidden layer features')
        plt.legend(loc="upper right")

        plt.savefig(args.save_path + "/" + str(idx+1) + ".png")

        plt.clf()
        plt.close()
        gc.collect()

    print("finished")


