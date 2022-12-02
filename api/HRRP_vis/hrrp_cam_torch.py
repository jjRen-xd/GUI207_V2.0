# -*- coding: utf-8 -*- #

import os
import cv2
import numpy as np
import torch
import scipy.io as scio
import argparse

from utils.CAM import t2n, HookValues, GradCAM, GradCAMpp, XGradCAM, EigenGradCAM, LayerCAM


name2label = {
    "DQ": 0,
    "DT": 1,
    "QDZ": 2,
    "XQ": 3,
    "Z": 4
}

def vis_cam(checkpoint_path, vis_layer, signal, label, name, method, top1 = False, gt_known = True):
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

    # backward
    if top1:
        pred_scores = logits.max(dim = 1)[0]
    elif gt_known:
        # GT-Known指标
        batch_size, _ = logits.shape
        _range = torch.arange(batch_size)
        pred_scores = logits[_range, label]
    else: 
        print("Error in indicator designation!!!")
        exit()
    # pred_labels = logits.argmax(dim = 1)
    model.zero_grad()                          
    pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)

    # Calculate CAM
    activations = hookValues.activations    # ([1, 15, 124, 1])
    gradients = hookValues.gradients        # ([1, 15, 124, 1])
    signal_array = t2n(signal.permute(0, 2, 3, 1))      # bz, nc, h, w -> bz, h, w, nc
    
    camCalculator = eval(method)(signal_array, [name])
    scaledCAMs = camCalculator(t2n(activations), t2n(gradients))    # bz, h, w (1, 512, 1)
    camsOverlay = camCalculator._overlay_cam_on_image(layerName="model."+vis_layer)

    return camsOverlay
        
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
    mat = scio.loadmat(matPath)
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
        '--cam_method',
        default="GradCAMpp",
        type=str,
        help='Visualization Algorithm Designation'
    )
    parser.add_argument(
        '--save_path',
        default="./figs/cam_output",
        type=str,
        help='The path of feature map to save'
    )
    args = parser.parse_args()

    # 读取数据
    signals, labels, labelName = read_mat(args.mat_path)
    signal = signals[args.mat_idx][None, None, :, None]
    label = labels[args.mat_idx] 

    # 计算CAM
    camsOverlay = vis_cam(
        args.checkpoint, 
        args.visualize_layer, 
        signal, 
        label, 
        labelName, 
        args.cam_method
    )

    # 保存图像
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    saveImgPath = args.save_path + "/"+args.cam_method+".png"
    cv2.imwrite(saveImgPath, camsOverlay[0])

    print("finished")


