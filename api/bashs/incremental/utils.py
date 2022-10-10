from scipy import optimize as op
import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from config import *


def getOneHot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot




def plot_tsne(features, label, title):
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=10)
    tsne_features = tsne.fit_transform(features)
    pre_len = len(label[label == 12])
    # tsne_features = tsne_features[:-pre_len, :]
    # tsne_label = label[:-pre_len,]
    pre_features = tsne_features[-pre_len:, :]
    tb_feature = pd.DataFrame(tsne_features, columns=['x1', 'x2'])
    tb_feature['label'] = label
    tb_feature.plot.scatter('x1', 'x2', c='label', colormap='jet')
    # plt.scatter(pre_features[:, 1], pre_features[:, 0], color='black')
    plt.title(title)
    plt.show()


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


def MinMaxUncertainty(model, exampler_data, exampler_label, memorylimit):
    data_list, label_list = [], []
    print("找最难样本：")
    for i in range(len(exampler_data)):
        data = torch.Tensor(exampler_data[i]).to(device)
        pred_i = model(data).cpu().detach().numpy()
        # pred_i, _ = model.predict([exampler_data[i], np.zeros((exampler_data[i].shape[0], 9))])
        pred_label = np.argmax(pred_i, axis=1)
        correct_index = []
        error_index = []
        # print("len_labeldtx:", len(exampler_data[i]))
        for j in range(len(exampler_data[i])):
            # 先从预测正确的样本中选
            if pred_label[j] == exampler_label[i][j]:
                correct_index.append(j)
            else:
                error_index.append(j)
        # 如果预测正确的样本不够memorylimit，就从错误的样本中选一部分
        if len(correct_index) < int(memorylimit / len(exampler_data)):
            gap_len = int(memorylimit / len(exampler_data)) - len(correct_index)
            gap_index = error_index[:gap_len]
            for n in range(len(gap_index)):
                correct_index.append(gap_index[n])
        correct_pred = pred_i[correct_index]
        # print(i, ":", correct_index)
        correct_data = exampler_data[i][correct_index]
        correct_label = exampler_label[i][correct_index]
        correct_pred = np.max(correct_pred, axis=1)
        correct_pred = np.argsort(correct_pred, axis=0)
        correct_pred = correct_pred[:int(memorylimit / (len(exampler_data)))]
        # correct_pred_2 = correct_pred[-int(memorylimit / (len(exampler_data) * 2)):]
        # correct_pred = np.concatenate([correct_pred_1, correct_pred_2], axis=0)
        # print("memory per class:", int(memorylimit / len(exampler_data)))
        correct_data = correct_data[correct_pred]
        correct_label = correct_label[correct_pred]
        # print("len_correct_data:", len(correct_data))
        data_list.append(correct_data)
        label_list.append(correct_label)

    # plot preserved data
    # fun = K.function([model.input], [model.layers[-2].output])
    feature_list = []
    for i in range(len(exampler_data)):
        data = torch.Tensor(exampler_data[i]).to(device)
        feature = model.get_feature(data).cpu().detach().numpy()
        # feature = fun(exampler_data[i])[0]
        feature_list.append(feature)
    features = np.concatenate(feature_list, axis=0)
    label = np.concatenate(exampler_label, axis=0)
    feature_list = []
    for i in range(len(data_list)):
        data = torch.Tensor(data_list[i]).to(device)
        feature = model.get_feature(data).cpu().detach().numpy()
        # feature = fun(data_list[i])[0]
        feature_list.append(feature)
    features_pre = np.concatenate(feature_list, axis=0)
    features = np.concatenate([features, features_pre], axis=0)
    label_pre = np.full(shape=(len(features_pre),), fill_value=12)
    label = np.concatenate([label, label_pre], axis=0)
    plot_tsne(features, label, 'tsne')

    return data_list, label_list


def mutualInfo_ori(model, exampler_data, exampler_label, memorylimit):
    # print(len(exampler_data), len(exampler_label))
    # print(exampler_data[0].shape, exampler_label[0].shape)
    data_len = int(memorylimit // len(exampler_data))
    data_list, label_list = [], []
    for i in range(len(exampler_data)):
        data = torch.Tensor(exampler_data[i]).to(device)
        pred_i = torch.softmax(model(data), dim=1).cpu().detach().numpy()
        pred_label = np.argmax(pred_i, axis=1)
        correct_index = []
        error_index = []
        for k in range(len(pred_label)):
            if pred_label[k] == exampler_label[i][k]:
                correct_index.append(k)
            else:
                error_index.append(k)
        if len(correct_index) < data_len:
            #print(data_len - len(correct_index))
            correct_index = np.concatenate([correct_index, error_index[:(data_len - len(correct_index))]], axis=0)
            #print(np.array(correct_index).shape)
        #print(exampler_data[i].shape)
        correct_index = np.array(correct_index).astype('int')
        correct_data = exampler_data[i][correct_index]
        correct_label = torch.Tensor(exampler_label[i][correct_index]).to(device)
        pred_i = pred_i[correct_index, :]
        label_hot = get_one_hot(correct_label, len(exampler_label)).cpu().numpy()
        info_list = []
        for j in range(len(pred_i)):
            # print("NMI:", pred_i[j,:], label_hot[j, :])
            mulInfo = metrics.normalized_mutual_info_score(pred_i[j, :], label_hot[j, :])
            # print(mulInfo)
            info_list.append(mulInfo)
        info_index = np.argsort(info_list)
        # print("number:", int(acc_list[i]*memorylimit))
        correct_label = correct_label.cpu()
        data_list.append(correct_data[info_index[:data_len]])
        label_list.append(correct_label[info_index[:data_len]])
    return data_list, label_list


def RandomPicking(exampler_data, exampler_label, memorylimit, random_seed=1):
    np.random.seed(random_seed)
    data_list, label_list = [], []
    for i in range(len(exampler_data)):
        if len(exampler_data[i]) < int(memorylimit // len(exampler_data)):
            index = np.random.choice(np.arange(len(exampler_data[i])), size=len(exampler_data[i]), replace=False)
        else:
            index = np.random.choice(np.arange(len(exampler_data[i])), size=int(memorylimit // len(exampler_data)),
                                     replace=False)
        data_list.append(exampler_data[i][index])
        label_list.append(exampler_label[i][index])

    return data_list, label_list


def get_weight_by_linearProgram(oldfeature, newfeature, weight, limit, k, feature_dim=512):
    print('get_weight_by_linearProgram...')
    # if old feature is 400 examples feature
    A_ub_oldfeature = oldfeature
    # print('A_ub_oldfeature shape: ' + str(A_ub_oldfeature.shape))
    neg_newfeature = np.zeros(oldfeature.shape)
    f_zero = np.zeros(oldfeature.shape)
    one_zero = np.zeros((1, feature_dim))
    if k != 1:
        # print('A_ub_newfeature shape: '+ str(newfeature.shape))
        # 纵向约束
        A_ub_newfeatures = []
        for i in range(k):
            A_ub_newfeature = neg_newfeature - newfeature[i].reshape(1, feature_dim)
            A_ub_newfeatures.append(A_ub_newfeature)
        fn_i = []
        for i in range(k):
            A_ubs = []
            for j in range(i):
                A_ubs.append(f_zero)
            A_ubs.append(A_ub_newfeatures[i])
            for m in range(k - i - 1):
                A_ubs.append(f_zero)
            A_ubs = np.concatenate(A_ubs, axis=1)
            fn_i.append(A_ubs)
        fn = np.concatenate(fn_i, axis=0)
        # 未知量比较约束
        fn_fn = []
        for i in range(k):
            fn_fn_i = []
            for j in range(k - 1):
                fn_i = []
                if j < i:
                    for m in range(j):
                        fn_i.append(one_zero)
                    fn_i.append(newfeature[i].reshape(1, -1))
                    for m in range(i - j - 1):
                        fn_i.append(one_zero)
                    fn_i.append(one_zero - newfeature[i].reshape(1, -1))
                    for m in range(k - i - 1):
                        fn_i.append(one_zero)
                else:
                    for m in range(i):
                        fn_i.append(one_zero)
                    fn_i.append(one_zero - newfeature[i].reshape(1, -1))
                    for m in range(j - i):
                        fn_i.append(one_zero)
                    fn_i.append(newfeature[i].reshape(1, -1))
                    for m in range(k - j - 2):
                        fn_i.append(one_zero)
                fn_i = np.concatenate(fn_i, axis=1)
                fn_fn_i.append(fn_i)
            fn_fn_i = np.concatenate(fn_fn_i, axis=0)
            fn_fn.append(fn_fn_i)
        fn_fn = np.concatenate(fn_fn, axis=0)
        # 横向约束
        fi_zero = []
        for l in range(k):
            fi = []
            for p in range(l):
                fi.append(neg_newfeature)
            fi.append(oldfeature)
            for p in range(k - l - 1):
                fi.append(neg_newfeature)
            fi = np.concatenate(fi, axis=1)
            fi_zero.append(fi)
        fi_zero = np.concatenate(fi_zero, axis=0)
        A_ub = np.concatenate((fn, fn_fn, fi_zero), axis=0)

        fni = []
        for i in range(k):
            fni.append(np.dot((one_zero - newfeature[i].reshape(1, -1)), weight).reshape(-1, ))
        fni = np.concatenate(fni, axis=0)

        zeroi = []
        for i in range(k * (k - 1)):
            zeroi.append(np.zeros((1)))
        zeroi = np.concatenate(zeroi, axis=0)

        fi_i = []
        for i in range(k):
            fi_i.append(np.dot(oldfeature, weight).diagonal())
        fi_i = np.concatenate(fi_i, axis=0)
        B_ub = np.concatenate((fni, zeroi, fi_i), axis=0)

        c = []
        for i in range(k):
            c.append(newfeature[i])
        c = np.concatenate(c, axis=0).reshape(-1, )

        bounds = np.zeros((feature_dim * k, 2))
        bounds[:, 0] += -limit
        bounds[:, 1] += limit

        res = op.linprog(-c, A_ub, B_ub, bounds=bounds)
        return res

    examplepreclass = int(oldfeature.shape[0] / weight.shape[1])
    c = newfeature.reshape(feature_dim, )
    A_ub = oldfeature
    bigweight = np.zeros(oldfeature.shape)
    # for i in range(0):
    #     bigweight[i*examplepreclass:i*examplepreclass+examplepreclass]=weight.T[i]
    bigweight = bigweight.T
    # print("bigweight:", bigweight.shape)
    # 对应位置相乘  查看矩阵对角线上的元素
    B_ub = np.dot(oldfeature, bigweight).diagonal()
    # print("B_ub:", B_ub)
    # print('B_ub shape: ', B_ub.shape, 'A_ub.shape', A_ub.shape)
    # print('B_ub shape: '+ str(B_ub.shape))
    bounds = np.zeros((feature_dim, 2))
    bounds[:, 0] += -limit
    bounds[:, 1] += limit
    # (512,1) (7,512) ()
    res = op.linprog(-c, A_ub, B_ub, bounds=bounds)

    return res


def get_eachClass_acc(y_pred, y_true, num_class):
    # Y_test_num = np.argmax(y_true, axis=1)
    Y_test_num = y_true
    Y_predict_num = np.argmax(y_pred, axis=1)
    class_count = [0.0 for _ in range(num_class)]
    class_acc_num = [0.0 for _ in range(num_class)]
    total_acc_num = 0.0
    for i in range(len(Y_test_num)):
        class_count[Y_test_num[i]] += 1
        if Y_test_num[i] == Y_predict_num[i]:
            class_acc_num[Y_test_num[i]] += 1
            total_acc_num += 1
    acc_ratio = [0.0 for _ in range(num_class)]
    for i in range(num_class):
        acc_ratio[i] = class_acc_num[i] / class_count[i]
        # print(i, ':', acc_ratio[i])
    return acc_ratio


def ConfusionMatrix(true, pred_label):
    matrix = confusion_matrix(true, pred_label)
    print("matrix@@@@@@@@",matrix)

# 绘制混淆矩阵
def show_confusion_matrix(classes, confusion_matrix, work_dir):
    plt.figure()
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = [] #百分比(行遍历)
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)   # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    config = {"font.family": 'Times New Roman'} 
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues) 
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = confusion_matrix.max() / 2.

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if i == j:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='red',
                     weight=5)  
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='red')
        else:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'/confusion_matrix.jpg', dpi=300)
    # plt.show()
