import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler


# 导入npz数据
def data_load(path):
    data = np.load(path)
    train_x = data['train_X']
    train_y = data['train_Y']
    test_x = data['test_X']
    test_y = data['test_Y']

    return train_x, train_y, test_x, test_y


# 归一化
def trans_norm(data):
    data_trans = list(map(list, zip(*data)))
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(np.array(data_trans))
    trans_data = np.array(list(map(list, zip(*data_norm))))

    return trans_data


# 画图
def show_pic(Y_pred, Y_test):
    x = np.array(len(Y_test))
    plt.plot(x, Y_test, color="blue")
    plt.plot(x, Y_pred, color="red")
    plt.show()


# 绘制混淆矩阵
def show_confusion_matrix(work_dir, classes, confusion_matrix):
    # classes = ['1', '2', '3', '4', '5', '6', '7', '8']
    # confusion_matrix 为分类的特征矩阵
    plt.figure()
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    # print(pshow)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12, rotation=20)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if i == j:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='red',
                     weight=5)  # 显示对应的数字
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='red')
        else:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'./confusion_matrix.jpg', dpi=300)
    #plt.show()


# 准确率
def show_acc(data):
    x = np.arange(data.shape[1])
    plt.plot(x, data)
    plt.show()


# 损失
def show_loss(data):
    x = np.arange(len(data))
    plt.plot(x, data)
    plt.savefig('./result/loss.jpg', dpi=300)
    plt.show()


# 损失
def show_loss_fl(data, name):
    plt.figure()
    x = np.arange(len(data))
    plt.plot(x, data)
    plt.savefig('./result/loss_'+str(name)+'.jpg', dpi=300)


# 归一到神经网络的最值
def norm_net(net_data, fea_data):
    net_max, net_min = np.max(net_data), np.min(net_data)
    minmax = MinMaxScaler(feature_range=[net_min, net_max])
    X = minmax.fit_transform(fea_data)
    return X


# 对应样本归一
def norm_one(net_data, fea_data):
    new_data = []
    for i in range(0, len(net_data)):
        net_one_max, net_one_min = np.max(net_data[i]), np.min(net_data[i])
        minmax = MinMaxScaler(feature_range=[net_one_min, net_one_max])
        X = minmax.fit_transform(fea_data[i].reshape(len(fea_data[i]), 1))
        new_one = []
        for j in range(0, len(X)):
            new_one.append(X[j][0])
        new_data.append(new_one)
    return np.array(new_data)
