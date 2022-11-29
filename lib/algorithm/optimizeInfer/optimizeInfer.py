import cv2,os,time
import torch.nn as nn
import argparse
from configs import cfgs
from dataset.HRRP.hrrp import *
# from dataset.RML2016 import RMLDataset
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.autograd import Variable
from utils.CAM import compute_gradcampp
from utils.strategy import step_lr, accuracy, drawConfusionMatrix
from utils.signal_vis import ave_mask_cam, t2n


parser = argparse.ArgumentParser(description='Evaluate by all dataset')
parser.add_argument('--choicedDatasetPATH', help='the directory of the dataset',default="D:/lyh/GUI207_V2.0/db/datasets/HRRP_simulate_128xN_c6")
parser.add_argument('--choicedModelPATH', help="the model path", default="D:/lyh/GUI207_V2.0/lib/algorithm/optimizeInfer/checkpoints/MsmcNet_hrrp_-15db_after.pth")
parser.add_argument('--choicedMatPATH', help="the .mat file path", default="D:/lyh/GUI207_V2.0/db/datasets/HRRP_simulate_128xN_c6/DT/DT.mat")
parser.add_argument('--choicedSampleIndex', help="the sampleIndex path",type=int, default=0)

parser.add_argument('--inferMode', help="infer mode(sample or dataset)", required=True)
args = parser.parse_args()
folder_name = []


class RMLDataset(Dataset):
    ''' 定义RMLDataset类，继承Dataset方法，并重写__getitem__()和__len__()方法 '''
    def __init__(self, data_root, data_label, transform=None):
        ''' 初始化函数，得到数据 '''
        self.data = data_root
        self.label = data_label
        self.transform = transform

    def __getitem__(self, index):
        ''' index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回 '''
        data = self.data[index]
        labels = self.label[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, labels

    def __len__(self):
        ''' 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼 '''
        return len(self.data)
def softmax(x):
    x -= np.max(x, axis = 0, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis = 0, keepdims = True)
    return x
# x_train, y_train, x_test, y_test = loadNpz()
# print("AAAAAAAA",x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# train_X, train_Y, test_X, test_Y, len_folder_name, folder_name=loadAllMat("D:/lyh/GUI207_V2.0/db/datasets/HRRP_simulate_128xN_c6")
# print("BBBBBBBB",train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)

# print("AAAAAAAA",type(x_train[0,0,0,0]))
# print("BBBBBBBB",type(train_X[0,0,0,0]))

# print("AAAAAAAA",type(y_test[2]))
# print("BBBBBBBB",type(test_Y[2]))



def eval_mask(model, dataloader_valid, criterion):
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
        drawConfusionMatrix(folder_name, args.choicedModelPATH, output_val.data, target_val.data, topk=(1,))
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1

# validation
def eval_base(model, dataloader_valid, criterion):
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
        drawConfusionMatrix(folder_name, args.choicedModelPATH, output_val.data, target_val.data, topk=(1,))

        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1


def inferMain():
    # Dataset
    x_train, y_train, x_test, y_test=loadAllMat(args.choicedDatasetPATH)
    #x_train, y_train, x_test, y_test=loadNpz()

    Dataset = RMLDataset

    model = torch.load(args.choicedModelPATH)
    model.cuda()
    # loss
    criterion = nn.CrossEntropyLoss().cuda()  # 交叉熵损失

    # DataLoader
    train_dataset = Dataset(x_train, y_train)  # RML2016.10a数据集
    dataloader_train = DataLoader(train_dataset, \
                                  batch_size=cfgs.batch_size, \
                                  num_workers=cfgs.num_workers, \
                                  shuffle=True, \
                                  drop_last=False)
    valid_dataset = Dataset(x_test, y_test)
    dataloader_valid = DataLoader(valid_dataset, \
                                  batch_size=cfgs.batch_size, \
                                  num_workers=cfgs.num_workers, \
                                  shuffle=True, \
                                  drop_last=False)

    val_loss, val_top1 = eval_mask(model, dataloader_valid, criterion)
    val_top1.cpu()
    #val_acc = val_top1[0].data.float()
    print('val_acc${:.3f}$'.format(val_top1[0].data.float()))
    # return val_acc.cpu().float()

def inferSample():
    className = args.choicedMatPATH.split("/")[-2]
    model = torch.load(args.choicedModelPATH)
    model.cuda()

    input_val = loadSample(args.choicedMatPATH, args.choicedSampleIndex)
    input_val = torch.Tensor(input_val).cuda()

    target_val = torch.full([1], folder_name.index(className), dtype=torch.int64).cuda()
    
    # print(type(input_val))
    # print(input_val.shape)
    # print(target_val.shape)
    # print(target_val[0],target_val.dtype)
    
    idx = 0
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
    T1 = time.perf_counter()
    output_val = model(input_val, cams_val, S_masks_val, B_masks_val)
    T2 = time.perf_counter()
    degrees=output_val.cpu().detach().numpy()[0]
    #degrees=softmax(degrees)
    for i in degrees:
        print(i,"$",end="")
    maxk = max((1,))
    _, pred = output_val.topk(maxk, 1, True, True) # [110,1]
    pred = pred.t().reshape((-1)).cpu().detach().numpy()
    print(pred[0],"$",end="")
    print("%.5f $" % (T2-T1),end="")

if __name__ == '__main__':
    #try:
    file_name = os.listdir(args.choicedDatasetPATH)  # 读取所有文件夹，将文件夹名存在列表中
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(args.choicedDatasetPATH+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序

    if(args.inferMode=="dataset"):
        inferMain()
        print("finished")
    elif(args.inferMode=="sample"):
        inferSample()
        print("\nfinished")
    # except:
    #     print("error")
    