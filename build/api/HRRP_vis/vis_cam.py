# -*- coding: utf-8 -*- #

import os
import cv2
import numpy as np
import torch

import argparse

from utils.CAM import t2n, HookValues, GradCAM, GradCAMpp, XGradCAM, EigenGradCAM, LayerCAM


name2label = {
    "DQ": 0,
    "DT": 1,
    "QDZ": 2,
    "XQ": 3,
    "Z": 4
}

def vis_cam(checkpoint_path, vis_layer, signal, name, method, top1 = False, gt_known = True):
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

    # backward
    if top1:
        pred_scores = logits.max(dim = 1)[0]
    elif gt_known:
        # GT-Known指标
        batch_size, _ = logits.shape
        _range = torch.arange(batch_size)
        pred_scores = logits[_range, name2label[name[0]]]
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
    
    camCalculator = eval(method)(signal_array, name)
    scaledCAMs = camCalculator(t2n(activations), t2n(gradients))    # bz, h, w (1, 512, 1)
    camsOverlay = camCalculator._overlay_cam_on_image(layerName="model."+vis_layer)

    return camsOverlay
        



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
        '--cam_method',
        default="GradCAMpp",
        type=str,
        help='Visualization Algorithm Designation'
    )
    parser.add_argument(
        '--save_path',
        default="/media/z840/HDD_1/LINUX/DD_vis/HRRP_vis/figs/cam_output",
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

    # 计算CAM
    camsOverlay = vis_cam(args.checkpoint, args.visualize_layer, signals, CALSSES, args.cam_method)

    # 保存图像
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    saveImgPath = args.save_path + "/"+"CAM_output.png"
    cv2.imwrite(saveImgPath, camsOverlay[0])

    print("finished")


