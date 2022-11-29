# 原网络
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from dataset.HRRP.hrrp import loadMat,loadNpz
from dataset.ACARS import ACARSDataset, loadNpy_acars
from networks.MsmcNet import MsmcNet_RML2016, MsmcNet_ACARS, MsmcNet_HRRP256
from utils.strategy import step_lr, accuracy
from utils.plot import draw_curve
from utils.CAM import compute_gradcampp
from utils.signal_vis import showOriSignal, showCamSignal, ave_mask_cam, t2n


def train():
    ''' 信号调制分类训练主程序 '''

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

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log_ori.txt', 'a')
    log.write('-' * 30 + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '-' * 30 + '\n')
    log.write(
        'model:{}\ndataset_name:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nsignal_len:{}\niter_smooth:{}\n'.format(
            cfgs.model, cfgs.dataset_name, cfgs.num_classes, cfgs.num_epochs,
            cfgs.lr, cfgs.signal_len, cfgs.iter_smooth))

    # load checkpoint
    if cfgs.resume:
        model = torch.load(os.path.join('./checkpoints', cfgs.checkpoint_name1))
        model_keys = model.state_dict()  # TODO:删除fc0.weight/fc0.bias
        print('yes')

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
    min_val = 0
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
            target = Variable(label).cuda().long()  # 此处target是真实标签,RML用.long();HRRP用.float()
            t = t + 1
            S_masks_list = []
            B_masks_list = []
            print('第{}批数据'.format(t))
            # [x] 第一次前向传播，目的是为了捕获梯度和激活，计算CAM

            output = model(input)  # inference

            # [ ] loss
            loss = criterion(output, target)  # 计算交叉熵损失,[110,11]与[110,]
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
            if val_top1[0].data > min_val:
                min_val = val_top1[0].data
                print('saving model')
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model, '{}/{}'.format('checkpoints', cfgs.checkpoint_name1))
            # if val_top1[0].data > max_val_acc:
            #    max_val_acc = val_top1[0].data
            #    print('Taking snapshot...')
            #    if not os.path.exists('./checkpoints'):
            #       os.makedirs('./checkpoints')
            #    torch.save(model, '{}/{}'.format('checkpoints', cfgs.checkpoint_name1))

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

        output_val = model(input_val)  # cams_val,S_masks_val,B_masks_val)
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