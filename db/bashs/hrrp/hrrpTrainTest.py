# encoding: utf-8
from gc import callbacks
import os
import argparse
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
from sklearn.metrics import classification_report

import sys, os
sys.path.append("..")
# from model_convert.hdf52trt import convert_hdf5_to_trt
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data',default="D:/lyh/GUI207_V2.0/db/datasets/falseHRRPmat_1x128")
parser.add_argument('--batch_size', help='the number of batch size',default=32)
parser.add_argument('--max_epochs', help='the number of epochs',default=4)


args = parser.parse_args()


# ���ݹ�һ��
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


# ��.mat�ļ���ȡ���ݲ�Ԥ����
def read_mat(read_path):
    # ��ȡ·���������ļ��е����Ʋ�����
    folder_path = read_path  # �����ļ�������·��
    file_name = os.listdir(folder_path)  # ��ȡ�����ļ��У����ļ����������б���
    folder_name = []
    for i in range(0, len(file_name)):
        # �ж��ļ������ļ�
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # ���ļ�������������

    # ��ȡ�����ļ����µ�����
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # ��ȡ����ļ����µ�.mat�ļ�����
        class_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[0]  # ����.mat�ļ�·��
        matrix_base = os.path.basename(class_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # ��ȡȥ����չ����.mat�ļ�����
        class_data = sio.loadmat(class_path)[matrix_name].T  # ����.mat�ļ�����ת��

        class_data_normalization = data_normalization(class_data)  # ��һ������

        # ���ݸ���64��
        class_data_picture = []
        for j in range(0, len(class_data_normalization)):
            class_data_one = class_data_normalization[j]
            empty = np.zeros((len(class_data_one), 64))
            for k in range(0, len(class_data_one)):
                empty[k, :] = class_data_one[k]
            class_data_picture.append(empty)
        class_data_picture = np.array(class_data_picture)  # �б�ת��Ϊ����

        # ���ñ�ǩ
        label = np.zeros((len(class_data_normalization), len(folder_name)))
        label[:, i] = 1

        # ����ѵ�����ݼ��Ͳ������ݼ�
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
    args.len = len(train_X)
    return train_X, train_Y, test_X, test_Y, len(folder_name), folder_name


# ��������洢
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


# ���ƻ�������
def show_confusion_matrix(classes, confusion_matrix):
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
    proportion = np.array(proportion).reshape(length, length)  # reshape(�еĳ��ȣ��еĳ���)
    pshow = np.array(pshow).reshape(length, length)
    config = {"font.family": 'Times New Roman'}  # ������������
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # ����������ʾ������
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = confusion_matrix.max() / 2.

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if i == j:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5)  # ��ʾ��Ӧ������
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # ��ʾ��Ӧ������
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.data_dir+'/confusion_matrix.jpg', dpi=300)
    # plt.show()


# ѵ��������׼ȷ������
def train_acc(epoch, acc):
    x = np.arange(epoch+1)[1:]
    plt.figure()
    plt.plot(x, acc)
    plt.scatter(x, acc)
    plt.grid()
    plt.title('Training accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.data_dir+'/training_accuracy.jpg', dpi=300)


# ��֤׼ȷ������
def val_acc(v_acc):
    x = np.arange(len(v_acc)+1)[1:]
    plt.figure()
    plt.plot(x, v_acc)
    plt.scatter(x, v_acc)
    plt.grid()
    plt.title('Verification accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Times', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.data_dir+'/verification_accuracy.jpg', dpi=300)



class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.total_iter_num = int(int(args.max_epochs)*int(args.len)/int(args.batch_size)+1)
        self.currrent_iter_num = 0
        self.timetaken = tf.timestamp().numpy()
        self.time_per_iter = 0
    def on_batch_begin(self,batch,logs = {}):
        if self.currrent_iter_num == 1:
            self.batchstarttime = tf.timestamp().numpy()
    def on_batch_end(self,batch,logs = {}):
        if self.currrent_iter_num == 1:
            self.batchendtime  = tf.timestamp().numpy()
            self.time_per_iter = self.batchendtime - self.batchstarttime
        if self.currrent_iter_num >= 1:
            print("RestTime:",(self.total_iter_num-self.currrent_iter_num)*self.time_per_iter)
            print("Schedule:", min(int(self.currrent_iter_num/self.total_iter_num*100),99))
        self.currrent_iter_num += 1



def run_main(x_train, y_train, x_test, y_test, class_num, folder_name):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_tensor=None,
                                                         input_shape=(x_train.shape[1], x_train.shape[2], 1),
                                                         pooling=None, classes=class_num))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='lr', patience=3, verbose=1,
                                                                   factor=0.5, min_lr=0.00001)
    # ��������ģ��
    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.data_dir+'/model_saving.hdf5', monitor='val_accuracy',
                                                    verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learning_rate_reduction, timecallback()]
    # model.summary()
    h = model.fit(x_train, y_train, batch_size=int(args.batch_size), epochs=int(args.max_epochs), shuffle=True,
              validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=2, validation_freq=1)
    h_parameter = h.history
    train_acc(int(args.max_epochs), h_parameter['accuracy'])
    val_acc(h_parameter['accuracy'])
    save_model = tf.keras.models.load_model(args.data_dir+'/model_saving.hdf5')
    Y_test = np.argmax(y_test, axis=1)
    #y_pred = save_model.predict_classes(x_test)
    y_pred = np.argmax(save_model.predict(x_test), axis=1)
    labels = folder_name  # ��ǩ��ʾ
    characteristic_matrix, accuracy_every_class = storage_characteristic_matrix(y_pred, Y_test, class_num)
    show_confusion_matrix(labels, characteristic_matrix)
    print(classification_report(Y_test, y_pred))


if __name__ == '__main__':
    # print("Train Starting")
    x_train, y_train, x_test, y_test, class_num, folder_name = read_mat(args.data_dir)
    run_main(x_train, y_train, x_test, y_test, class_num, folder_name)
    print("Train Ending")
    # convert_hdf5_to_trt(args.data_dir+'/model_saving.hdf5')
