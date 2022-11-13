import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler


# 从.mat文件读取数据并预处理
def read_mat(read_path):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]) and file_name[i]!="model_saving":
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序

    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的.mat文件名称
        class_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[0]  # 类别的.mat文件路径
        matrix_base = os.path.basename(class_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        class_data = sio.loadmat(class_path)[matrix_name].T  # 读入.mat文件，并转置

        # 设置标签
        label = np.zeros((len(class_data), len(folder_name)))
        label[:, i] = 1

        # 划分训练数据集和测试数据集
        x_train = class_data[:int(len(class_data)/2), :]
        x_test = class_data[int(len(class_data)/2):, :]
        y_train = label[:int(len(class_data)/2), :]
        y_test = label[int(len(class_data)/2):, :]

        if i == 0:
            train_x = x_train
            test_x = x_test
            train_y = y_train
            test_y = y_test
        else:
            train_X = np.concatenate((train_x, x_train))
            train_Y = np.concatenate((train_y, y_train))
            test_X = np.concatenate((test_x, x_test))
            test_Y = np.concatenate((test_y, y_test))

            train_x = train_X
            train_y = train_Y
            test_x = test_X
            test_y = test_Y

    return train_X, train_Y, test_X, test_Y, folder_name


# 数据归一化
def data_normalization(data):
    minmax = MinMaxScaler()
    X = minmax.fit_transform(data)
    return X


# 特征矩阵存储
def storage_characteristic_matrix(result, test_Y, output_size):
    characteristic_matrix = np.zeros((output_size, output_size))
    for i in range(0, len(result)):
        characteristic_matrix[test_Y[i], result[i]] += 1
    single_class_sum = np.zeros(output_size)
    pre_right_sum = np.zeros(output_size)
    for i in range(0, output_size):
        single_class_sum[i] = sum(characteristic_matrix[i])
        pre_right_sum[i] = characteristic_matrix[i, i]
    accuracy_every_class = pre_right_sum/single_class_sum
    accuracy = sum(pre_right_sum)/sum(single_class_sum)
    return characteristic_matrix, accuracy_every_class, accuracy


# 绘制混淆矩阵
def show_confusion_matrix(classes, confusion_matrix, path):
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
    plt.savefig(path + '/confusion_matrix.jpg', dpi=300)
    # plt.show()


# 绘制features_Accuracy
def show_feature_selection(ac_score_list, feature_start, feature_end, feature_interval, path):
    plt.figure(figsize=(11, 5))
    x_interval = [i for i in range(feature_start, feature_end+1, feature_interval)]
    plt.plot(x_interval, ac_score_list)
    plt.scatter(x_interval, ac_score_list)
    plt.grid()
    # plt.title(dataset_name)
    plt.xticks(x_interval)
    plt.xlabel("Number of features used")
    plt.ylabel("Accuracy")
    plt.savefig(path + '/features_Accuracy.jpg', dpi=400)
    # plt.show()

