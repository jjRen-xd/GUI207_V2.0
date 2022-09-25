# encoding: utf-8
from gc import callbacks
import os
import argparse
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
# import keras as K
# from tensorflow import keras as K
# import re
# from functools import reduce
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from matplotlib import rcParams
# from datetime import datetime
from sklearn.metrics import classification_report
import sys

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--data_dir', help='the directory of the training data',default="../../db/datasets/falseHRRPmat_1x128")
    parser.add_argument('--time', help='the directory of the training data',default="2022-09-21-21-52-17")
    parser.add_argument('--work_dir', help='the directory of the training data',default="../../db/trainLogs")
    parser.add_argument('--model_name', help='the directory of the training data',default="model")
    parser.add_argument('--batch_size', help='the number of batch size',default=32)
    parser.add_argument('--max_epochs', help='the number of epochs',default=4)

    args = parser.parse_args()
    return args


# 数据归一化
def data_normalization(data):
    DATA = []
    for i in range(0, len(data)):
        data_max = max(data[i])
        data_min = min(data[i])
        data_norm = []
        for j in range(0, len(data[i])):
            data_one = (data[i][j] - data_min) / (data_max - data_min)
            data_norm.append(data_one)
        DATA.append(data_norm)
    DATA = np.array(DATA)
    return DATA


# 从.mat文件读取数据并预处理
def read_mat(read_path):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序

    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的.mat文件名称
        class_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[0]   # 类别的.mat文件路径
        matrix_base = os.path.basename(class_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        class_data = sio.loadmat(class_path)[matrix_name].T  # 读入.mat文件，并转置

        class_data_normalization = data_normalization(class_data)  # 归一化处理

        # 数据复制64次
        class_data_picture = []
        for j in range(0, len(class_data_normalization)):
            class_data_one = class_data_normalization[j]
            empty = np.zeros((len(class_data_one), 64))
            for k in range(0, len(class_data_one)):
                empty[k, :] = class_data_one[k]
            class_data_picture.append(empty)
        class_data_picture = np.array(class_data_picture)   # 列表转换为数组

        # 设置标签
        label = np.zeros((len(class_data_normalization), len(folder_name)))
        label[:, i] = 1

        # 划分训练数据集和测试数据集
        x_train = class_data_picture[:int(len(class_data_picture)/2), :]
        x_test = class_data_picture[int(len(class_data_picture)/2):, :]
        y_train = label[:int(len(class_data_picture)/2), :]
        y_test = label[int(len(class_data_picture)/2):, :]
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
    # args.len = len(train_X)
    return train_X, train_Y, test_X, test_Y, len(folder_name), folder_name


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
    return characteristic_matrix, accuracy_every_class


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
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5)  
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'/confusion_matrix.jpg', dpi=300)
    # plt.show()


# 训练过程中准确率曲线
def train_acc(epoch, acc, work_dir):
    x = np.arange(epoch+1)[1:]
    plt.figure()
    plt.plot(x, acc)
    plt.scatter(x, acc)
    plt.grid()
    plt.title('Training accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'/training_accuracy.jpg', dpi=300)


# 验证准确率曲线
def val_acc(v_acc, work_dir):
    x = np.arange(len(v_acc)+1)[1:]
    plt.figure()
    plt.plot(x, v_acc)
    plt.scatter(x, v_acc)
    plt.grid()
    plt.title('Verification accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Times', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'/verification_accuracy.jpg', dpi=300)


def run_main(x_train, y_train, x_test, y_test, class_num, folder_name, work_dir, model_name):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_tensor=None,input_shape=(x_train.shape[1], x_train.shape[2], 1),pooling=None, classes=class_num))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='lr', patience=3, verbose=1,factor=0.5, min_lr=0.00001)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(work_dir+'/model/'+model_name+'.hdf5', monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learning_rate_reduction]
    # model.summary()
    h = model.fit(x_train, y_train, batch_size=int(args.batch_size), epochs=int(args.max_epochs), shuffle=True,
              validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=2, validation_freq=1)
    h_parameter = h.history
    train_acc(int(args.max_epochs), h_parameter['accuracy'], work_dir)
    val_acc(h_parameter['accuracy'], work_dir)
    save_model = tf.keras.models.load_model(work_dir+'/model/'+model_name+'.hdf5')
    Y_test = np.argmax(y_test, axis=1)
    #y_pred = save_model.predict_classes(x_test)   
    y_pred = np.argmax(save_model.predict(x_test), axis=1)
    labels = folder_name 
    characteristic_matrix, accuracy_every_class = storage_characteristic_matrix(y_pred, Y_test, class_num)
    show_confusion_matrix(labels, characteristic_matrix, work_dir)
    print(classification_report(Y_test, y_pred))


if __name__ == '__main__':
    try:
        args = parse_args()
        x_train, y_train, x_test, y_test, class_num, folder_name = read_mat(args.data_dir)
        datasetName = args.data_dir.split("/")[-1].split("_")[0]
        args.work_dir = args.work_dir+'/'+args.time+'-HRRP-'+datasetName
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
            os.makedirs(args.work_dir + '/model')

        run_main(x_train, y_train, x_test, y_test, class_num, folder_name, args.work_dir, args.model_name)

        # os.system("python ../../api/bashs/hdf52trt.py --model_type HRRP --work_dir "+args.work_dir+" --model_name "+args.model_name)
        print("Train Ended:")
    except Exception as re:
        print("Train Failed:",re)
