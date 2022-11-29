# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/train.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — —
                            --> DD信号识别（可解释）系列代码 <--
                    -- 训练主程序，移植之前信号识别tensorflow代码至PyTorch，
                    并进行项目工程化处理
                    -- TODO train()部分代码需要模块化，特别是指标记录、数据集
                    方面
                    — — — — — — — — — — — — — — — — — — — — — — — — — — —
# Module called:    <0> PATH_ROOT/configs.py
                    <1> PATH_ROOT/dataset/RML2016.py
                    <2> PATH_ROOT/networks/MsmcNet.py
                    <3> PATH_ROOT/utils/strategy.py;plot.py
                    <4> PATH_ROOT/dataset/ACARS.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — —
# Function List:    <0> train():
                        -- 训练主程序，包含了学习率调整、log记录、收敛曲线绘制
                        ，每训练n(1)轮验证一次，保留验证集上性能最好的模型
                    <1> eval():
                        -- 验证当前训练模型在测试集中的性能4
                    — — — — — — — — — — — — — — — — — — — — — — — — — — —
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   <0> | JunJie Ren |   v1.0    | 2020/06/14 | 使用PyTorch复现之前keras代码
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   <1> | JunJie Ren |   v1.1    | 2020/07/09 |    新增ACARS训练程序选项
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   <2> |
--------------------------------------------------------------------------

'''

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from configs import cfgs
from dataset.RML2016 import RMLDataset, loadNpy
from dataset.HRRP.hrrp import loadNpz
from networks.MsmcNet import MsmcNet_RML2016, MsmcNet_ACARS, MsmcNet_HRRP256
from utils.strategy import step_lr, accuracy
from utils.plot import draw_curve
from utils.CAM import compute_gradcampp
from utils.signal_vis import showOriSignal, showCamSignal, ave_mask_cam, t2n


def train():
    ''' 信号调制分类训练主程序 '''
    # model
    # if cfgs.model == "MsmcNet_RML2016":
    #    model = MsmcNet_RML2016(num_classes=cfgs.num_classes)
    # elif cfgs.model == "MsmcNet_ACARS":
    #    model = MsmcNet_ACARS(num_classes=cfgs.num_classes)
    # elif cfgs.model == 'MsmcNet_HRRP':
    #    model = MsmcNet_RML2016(num_classes=cfgs.num_classes)
    # else :
    #      print('ERROR: No model {}!!!'.format(cfgs.model))
    # print(model)
    # '''model = torch.nn.DataParallel(model)    # 多卡预留'''
    # model.cuda()

    # Dataset
    if cfgs.dataset_name == "RML2016.04c":
        x_train, y_train, x_test, y_test = loadNpy(
            cfgs.train_path,
            cfgs.test_path,
            cfgs.process_IQ
        )
        Dataset = RMLDataset
    elif cfgs.dataset_name == "hrrp":
        x_train, y_train, x_test, y_test = loadNpz()
        Dataset = RMLDataset
        if x_train.shape[2] == 128:
            model = MsmcNet_RML2016(num_classes=cfgs.num_classes)
        else:
            model = MsmcNet_HRRP256(num_classes=cfgs.num_classes)
        model.cuda()
    else:
        print('ERROR: No Dataset {}!!!'.format(cfgs.model))
    # BUG,BUG,BUG,FIXME
    transform = transforms.Compose([
        # transforms.ToTensor()
        # waiting add
    ])
    # Train data
    train_dataset = Dataset(x_train, y_train, transform=transform)  # RML2016.10a数据集
    dataloader_train = DataLoader(train_dataset, \
                                  batch_size=cfgs.batch_size, \
                                  num_workers=cfgs.num_workers, \
                                  shuffle=True, \
                                  drop_last=False)
    # Valid data
    valid_dataset = Dataset(x_test, y_test, transform=transform)
    dataloader_valid = DataLoader(valid_dataset, \
                                  batch_size=cfgs.batch_size, \
                                  num_workers=cfgs.num_workers, \
                                  shuffle=True, \
                                  drop_last=False)

    # # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')
    log.write('-' * 30 + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '-' * 30 + '\n')
    log.write(
        'model:{}\ndataset_name:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nsignal_len:{}\niter_smooth:{}\n'.format(
            cfgs.model, cfgs.dataset_name, cfgs.num_classes, cfgs.num_epochs,
            cfgs.lr, cfgs.signal_len, cfgs.iter_smooth))
    del_key = []
    # load checkpoint
    if cfgs.resume:
        model = torch.load(os.path.join('./checkpoints', cfgs.checkpoint_name1))
        model_weight = model.state_dict()
        for key, _ in model_weight.items():
            if 'fc1' in key:
                del_key.append(key)
        for key in del_key:
            del model_weight[key]
        model.load_state_dict(model_weight, strict=False)

    # loss
    criterion = nn.CrossEntropyLoss().cuda()  # 交叉熵损失

    # train
    sum = 0
    idx = 0
    train_loss_sum = 0
    train_top1_sum = 0
    max_val_acc = 0
    train_draw_acc = []
    val_draw_acc = []

    lr = cfgs.lr
    t = 0
    for epoch in range(cfgs.num_epochs):
        ep_start = time.time()

        # adjust lr
        # lr = half_lr(cfgs.lr, epoch)
        lr = step_lr(epoch, lr)

        # optimizer FIXME
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

        model.train()
        top1_sum = 0

        for i, (signal, label) in enumerate(dataloader_train):
            input = Variable(signal).cuda().float()  # [110,1,128,2]
            target = Variable(label).cuda().long()  # 此处target是真实标签
            t = t + 1
            S_masks_list = []
            B_masks_list = []
            print('第{}批数据'.format(t))
            # [x] 第一次前向传播，目的是为了捕获梯度和激活，计算CAM
            output = model(input)  # inference
            # loss = criterion(output, target) # 计算交叉熵损失，output是训练预测出的标签
            # loss.backward()                  # 反传 [x] 第一次需要反传吗 √CAM计算需要梯度
            # [x] CAM,FIXME：需要把前背景特征图在第一次前向传播时就算好?

            cams, scores, pred_labels, cam_h = compute_gradcampp(input, target, model, gt_known=True)
            for ipt, cam, tgt, _ in zip(input, cams, target, pred_labels):
                idx += 1
                # showImgSignal(t2n(image), label)              # 绘制处理后的二维信号
                # showCamSignal(t2n(ipt), cam, label, idx)    # 绘制CAM叠加图，IQ两路细粒度
                S_mask, B_mask = ave_mask_cam(t2n(ipt), cam, tgt, idx)  # 不用t2n(image),signal是tensor [256,2]
                S_mask = cv2.resize(S_mask, (1, 2 * cam_h), interpolation=cv2.INTER_AREA)
                B_mask = cv2.resize(B_mask, (1, 2 * cam_h), interpolation=cv2.INTER_AREA)  # [10,1]
                S_masks_list.append(S_mask)
                B_masks_list.append(B_mask)
                if idx % 100 == 0:
                    print('{} samples has done ave_mask'.format(idx))
                Sa_masks_list = np.array(S_masks_list)
                Ba_masks_list = np.array(B_masks_list)
            S_masks = torch.tensor(Sa_masks_list, requires_grad=True).unsqueeze(1).repeat(1, 15, 1, 1).to(
                torch.float32).cuda()  # [bz,15,5,1]
            B_masks = torch.tensor(Ba_masks_list, requires_grad=True).unsqueeze(1).repeat(1, 15, 1, 1).to(
                torch.float32).cuda()

            #   # [ ] 第二次前向传播，融合特征，优化网络参数
            output = model(input, cams, S_masks, B_masks)

            # [ ] loss
            loss = criterion(output, target)  # 计算交叉熵损失
            optimizer.zero_grad()
            loss.backward()  # 反传

            optimizer.step()
            
            top1 = accuracy(output.data, target.data, topk=(1,))  # 计算top1分类准确率
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += top1[0]
            sum += 1
            top1_sum += top1[0]

            if (i + 1) % cfgs.iter_smooth == 0:
                print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f'
                      % (epoch + 1, cfgs.num_epochs, i + 1, len(train_dataset) // cfgs.batch_size,
                         lr, train_loss_sum / sum, train_top1_sum / sum))
                log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f\n'
                          % (epoch + 1, cfgs.num_epochs, i + 1, len(train_dataset) // cfgs.batch_size,
                             lr, train_loss_sum / sum, train_top1_sum / sum))
                sum = 0
                train_loss_sum = 0
                train_top1_sum = 0

        train_draw_acc.append(top1_sum.cpu() / len(dataloader_train))

        epoch_time = (time.time() - ep_start) / 60.
        if epoch % cfgs.valid_freq == 0 and epoch < cfgs.num_epochs:
            # eval
            val_time_start = time.time()
            val_loss, val_top1 = eval(model, dataloader_valid, criterion)
            val_draw_acc.append(val_top1.cpu())
            val_time = (time.time() - val_time_start) / 60.

            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f'
                  % (epoch + 1, cfgs.num_epochs, val_loss, val_top1, val_time * 60, max_val_acc))
            print('epoch time: {}s'.format(epoch_time * 60))
            if val_top1[0].data > max_val_acc:
                max_val_acc = val_top1[0].data
                print('Taking snapshot...')
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model, '{}/{}'.format('checkpoints', cfgs.checkpoint_name2))

            log.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f\n'
                      % (epoch + 1, cfgs.num_epochs, val_loss, val_top1, val_time * 60, max_val_acc))
    draw_curve(train_draw_acc, val_draw_acc)
    log.write('-' * 40 + "End of Train" + '-' * 40 + '\n')
    log.close()


# validation
def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    idx = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda().float()
        target_val = Variable(label).cuda()
        S_masks_list_val = []
        B_masks_list_val = []
        cams_val, scores, pred_labels, cam_h = compute_gradcampp(input_val, target_val, model, gt_known=True)
        for ipt, cam, tgt, _ in zip(input_val, cams_val, target_val, pred_labels):
            idx += 1
            S_mask, B_mask = ave_mask_cam(t2n(ipt), cam, tgt, idx)  # 不用t2n(image),signal是tensor
            S_mask = cv2.resize(S_mask, (1, 2 * cam_h), interpolation=cv2.INTER_AREA)
            B_mask = cv2.resize(B_mask, (1, 2 * cam_h), interpolation=cv2.INTER_AREA)
            S_masks_list_val.append(S_mask)
            B_masks_list_val.append(B_mask)
            if idx % 800 == 0:
                print('{} val_samples has done ave_mask'.format(idx))
            Sa_masks_list = np.array(S_masks_list_val)
            Ba_masks_list = np.array(B_masks_list_val)
        S_masks_val = torch.tensor(Sa_masks_list, requires_grad=True).unsqueeze(1).repeat(1, 15, 1, 1).to(
            torch.float32).cuda()  # [bz,15,5,1]
        B_masks_val = torch.tensor(Ba_masks_list, requires_grad=True).unsqueeze(1).repeat(1, 15, 1, 1).to(
            torch.float32).cuda()
        output_val = model(input_val, cams_val, S_masks_val, B_masks_val)
        loss = criterion(output_val, target_val)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))

        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1


if __name__ == "__main__":
    train()