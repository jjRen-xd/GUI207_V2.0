# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/visualize.py
# Author:           JunJie Ren
# Version:          v1.1
# Created:          2021/06/15
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 利用GradCAM++可视化技术，解释网络隐层特征
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
        |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <0> | JunJie Ren |   v1.0    | 2020/06/15 |           creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    <1> | JunJie Ren |   v1.1    | 2020/07/09 |    新增ACARS可视化选项
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    <2> | JunJie Ren |   v1.2    | 2021/07/16 | 擦除CAM区域，计算可解释指标
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    <3> | JunJie Ren |   v1.3    | 2021/06/24 | 新增SigCAM的计算，失败
                                              | 增加导向反向传播可视化
--------------------------------------------------------------------------
'''

import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from configs import cfgs
from dataset.HRRP import HRRPDataset
from networks.hrrpCNN import CNN_HRRP512

from utils.strategy import accuracy
from utils.plot import plot_confusion_matrix
from utils.CAM import compute_gradcampp, compute_Sigcam
from utils.signal_vis import showImgSignal,showOriSignal,showCamSignal,mask_image,mask_image_hard
from utils.Guided_BP import GuidedBackprop


def showCAM(cam_thr, reserve_erase):
    # model
    model = torch.load(os.path.join('./checkpoints', cfgs.checkpoint_name))    # load checkpoint
    print(model)
    # model = torch.nn.DataParallel(model)  # 多卡预留
    model.cuda()
    
    # Dataset
    transform = transforms.Compose([ 
                                        # transforms.ToTensor()
                                        # waiting add
                                    ])
    # Valid data
    valid_dataset = HRRPDataset(cfgs.validTxtPath, transform=transform)  
    dataloader_valid = DataLoader(valid_dataset, \
                                batch_size=cfgs.batch_size, \
                                num_workers=cfgs.num_workers, \
                                shuffle=False, \
                                drop_last=False)

            
    # Valid data
    transform = transforms.Compose([ 
                                        # transforms.ToTensor()
                                        # waiting add
                                    ])
    idx = 0
    scores_drop = []    # 得分上升统计
    scores_incr = []    # 得分下降统计
    scores_same = []
    CAM_ERROR = []      # CAM 无法计算的label统计

    mask_top1_sum = 0
    ori_top1_sum = 0
    vis_layers = [model.Block_1[2], model.Block_2[2], model.Block_3[2], model.Block_4[2]]
    for layer_idx, vis_layer in enumerate(vis_layers):
        idx = 0
        for images, labels in dataloader_valid:
            images = Variable(images).cuda().float()
            labels = Variable(labels).cuda()
            # SigCAM
            # cams_target, cams_IQ, cams, scores, pred_labels = compute_Sigcam(images, labels, model, vis_layer, gt_known = True)     # (bz, 128, 2)
            # Grad-CAM++
            cams, scores, pred_labels, activations = compute_gradcampp(images, labels, model, vis_layer, gt_known = True)     # (bz, 128, 2)

            for image, cam, label, pred_label, activation in zip(images, cams, labels, pred_labels, activations):
                print(idx, layer_idx)
                idx += 1
                # showImgSignal(t2n(image), label)              # 绘制处理后的二维信号
                # showOriSignal(t2n(image), label, idx)         # 绘制原始信号图
                showOriSignal(t2n(activation), label, idx)         # 绘制原始信号图
                # showCamSignal(t2n(image), cam, label, idx+1, layer_idx+1)    # 绘制target_CAM叠加图
                # showCamSignal(t2n(image), cams_IQ[0], label, idx)        # 绘制IQ_CAM叠加图
                # showCamSignal(t2n(image), cam, label, idx)    # 绘制CAM叠加图，IQ两路细粒度
            
            break
        '''擦除CAM，重新计算得分变化，计算指标'''
            # masked_image = mask_image(cam, image, reserveORerase = reserve_erase)
            # # masked_image = mask_image_hard(cam, image, reserveORerase = reserve_erase, thr = cam_thr)
            # showCamSignal(t2n(masked_image), cam, label, idx)
            # softmax = nn.Softmax(dim=0)
            
            # score_ori = t2n(softmax(model(image.unsqueeze(0))[0])[label])

            # logti_masked = softmax(model(masked_image.unsqueeze(0))[0])
            # label_pred_masked = logti_masked.argmax(dim = 0)
            # score_masked = t2n(logti_masked[label])
            # print(score_masked, score_ori)
            
            # score_change = score_masked - score_ori
            # if score_change == 0:
            #     scores_same.append([score_masked, score_ori, score_change/score_ori])
            # elif score_change > 0:
            #     scores_incr.append([score_masked, score_ori, score_change/score_ori])
            # else:
            #     scores_drop.append([score_masked, score_ori, score_change/score_ori])

            # mask_top1_sum += 1 if label == label_pred_masked else 0
            # ori_top1_sum += 1 if label == pred_label else 0
            
            # if idx % 100 == 0:
            #     print("Have finished {} samples".format(idx))

    # mask_top1 = mask_top1_sum / idx
    # ori_top1 = ori_top1_sum / idx

    # print("\n****************** Complate {} THR, reserve_erase: {} ! *******************".format(cam_thr, "erase" if reserve_erase else "reserve"))
    
    # print("drop_num: {}, incr_num: {}, same_num: {}, ERROR_num: {}"\
    #     .format(len(scores_drop), len(scores_incr), len(scores_same), len(CAM_ERROR)))
    # scores_drop = np.array(scores_drop)
    # scores_incr = np.array(scores_incr)
    # print("scores_drop:", np.mean(scores_drop[:,-1]))
    # print("scores_incr:", np.mean(scores_incr[:,-1]))
    # print("mask_top1:{}, ori_top1:{}".format(mask_top1, ori_top1))

    # return len(scores_drop), len(scores_incr), len(scores_same), len(CAM_ERROR), np.mean(scores_drop[:,-1]), mask_top1


def showGBP():
    # model
    model = torch.load(os.path.join('./checkpoints', cfgs.checkpoint_name))    # load checkpoint
    print(model)
    # model = torch.nn.DataParallel(model)  # 多卡预留
    model.cuda()
    
   # Dataset
    if cfgs.dataset_name == "RML2016.04c":
        x_train, y_train, x_test, y_test = loadNpy(
            cfgs.train_path,
            cfgs.test_path,
            cfgs.process_IQ
        )
        Dataset = RMLDataset
    elif cfgs.dataset_name == "ACARS":
        x_train, y_train, x_test, y_test = loadNpy_acars(
            cfgs.train_path_x,
            cfgs.train_path_y,
            cfgs.test_path_x,
            cfgs.test_path_y,
            cfgs.process_IQ
        )
        Dataset = ACARSDataset
    else :
        print('ERROR: No Dataset {}!!!'.format(cfgs.model))
        
    # Valid data
    # BUG,BUG,BUG,FIXME
    transform = transforms.Compose([ 
                                        # transforms.ToTensor()
                                        # waiting add
                                    ])

    valid_dataset = Dataset(x_test, y_test, transform=transform)
    dataloader_valid = DataLoader(valid_dataset, \
                                batch_size=cfgs.batch_size, \
                                num_workers=cfgs.num_workers, \
                                shuffle=True, \
                                drop_last=False)

    for images, labels in dataloader_valid:
        images = Variable(images).cuda().float()
        labels = Variable(labels).cuda()
        # TODO 目前只支持batch=1的情况
        # 指导性反传
        GBP = GuidedBackprop(model, images)
        # 获取梯度
        guided_grads = GBP.generate_gradients(images, labels)   # FIXED np.array(1, 1, 4096, 2)
        showOriSignal(t2n(images[0]), labels[0], 1)         # 绘制原始信号图
        showOriSignal(guided_grads[0], labels[0], 1)         # 绘制显著性图
        cams, scores, pred_labels = compute_gradcampp(images, labels, model, gt_known = True)
        showCamSignal(guided_grads[0]*2, cams[0], labels[0], 1)
        ''' FOR IMAGE
        # 保存伪彩梯度图
        save_gradient_images(guided_grads, '_Guided_BP_color')
        # 转换到灰度空间
        grayscale_guided_grads = convert_to_grayscale(guided_grads)
        # 保存灰度梯度图
        save_gradient_images(grayscale_guided_grads,'_Guided_BP_gray')
        # 正负显著性图
        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        save_gradient_images(pos_sal, '_pos_sal')
        save_gradient_images(neg_sal, '_neg_sal')
        '''
        print('Guided backprop completed')
        # exit()


def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)


if __name__ == '__main__':
    ''' Guided BackProp '''
    # showGBP()

    ''' CAM only '''
    showCAM(cam_thr = 0.3, reserve_erase = 0)
    
    ''' CAM muti TEST '''
    # metrics_log = []
    # thrs = np.arange(0, 1, 0.05)
    # for thr in thrs:
    #     for erase in range(2):
    #         drop_num, incr_num, same_num, error, drop_rate, top1_acc = showCAM(cam_thr = thr, reserve_erase = erase)
    #         metrics_log.append([drop_num, incr_num, same_num, error, drop_rate, top1_acc])
    # metrics_log = np.array(metrics_log)
    # np.save("./log/metrics_CAM.npy", metrics_log)
