# -*- coding: UTF-8 -*-
from ast import arg
import os
import argparse
import time
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np
import afs_model
from data_process import read_mat, storage_characteristic_matrix
from data_process import show_feature_selection, show_confusion_matrix
from utils import BatchCreate
from sklearn.metrics import classification_report
from matplotlib import rcParams
tf.compat.v1.disable_eager_execution()


parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data')
parser.add_argument('--batch_size', help='the number of batch size')
parser.add_argument('--max_epochs', help='the number of epochs')


args = parser.parse_args()


class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.total_iter_num = int(int(args.max_epochs)*int(args.len)/int(args.batch_size)+1)
        self.currrent_iter_num = 0
        self.timetaken = time.time()
        self.time_per_iter = 0
    def on_batch_begin(self,batch,logs = {}):
        if self.currrent_iter_num == 1:
            self.batchstarttime = time.time()
    def on_batch_end(self,batch,logs = {}):
        if self.currrent_iter_num == 1:
            self.batchendtime  = time.time()
            self.time_per_iter = (self.batchendtime - self.batchstarttime)
        if self.currrent_iter_num >= 1:
            print("RestTime:",max(1,(self.total_iter_num*(40-args.modelid)-self.currrent_iter_num)*self.time_per_iter))
            print("Schedule:", min(args.schedule+int(self.currrent_iter_num/(self.total_iter_num*39)*100),99))
        self.currrent_iter_num += 1
    def on_train_end(self,batch,logs = {}):
        args.modelid += 1
        args.schedule = min(args.schedule+int(self.currrent_iter_num/(self.total_iter_num*39)*100),99)


def test(train_X, train_Y, test_X, test_Y, epoch, output_size, fea_num):
    train_model = keras.models.Sequential([
        keras.layers.Conv1D(64, kernel_size=1, padding='valid', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(output_size, activation='softmax')])
    train_model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    # 建立存储保存后的模型的文件夹
    if not os.path.exists(args.data_dir + '/model_saving/'):
        os.makedirs(args.data_dir + '/model_saving/')
    save_model_path = args.data_dir + '/model_saving/' + str(fea_num) + '.hdf5'
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.99, patience=3,
                                                             verbose=0, min_lr=0.0001)
    checkpoint = keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learn_rate_reduction, timecallback()]
    train_model.fit(train_X, train_Y, batch_size=int(args.batch_size), epochs=epoch, shuffle=True,
                   validation_data=(test_X, test_Y), callbacks=callbacks_list, verbose=0, validation_freq=1)
    test_model = keras.models.load_model(save_model_path)
    Y_test = np.argmax(test_Y, axis=1)
    Y_pred = test_model.predict_classes(test_X)

    return Y_test, Y_pred


def run_test(A, train_X, train_Y, test_X, test_Y, epoch, output_size, f_start, f_end, f_interval):
    all_characteristic_matrix = []  # 存储特征矩阵
    attention_weight = A.mean(0)  # [512,]
    AFS_wight_rank = list(np.argsort(attention_weight))[::-1]
    ac_score_list = []
    for K in range(f_start, f_end + 1, f_interval):
        use_train_x = np.expand_dims(train_X[:, AFS_wight_rank[:K]], axis=-1)
        use_test_x = np.expand_dims(test_X[:, AFS_wight_rank[:K]], axis=-1)
        label_class, predicted_class = test(use_train_x, train_Y, use_test_x, test_Y, epoch, output_size, K)
        characteristic_matrix, accuracy_every_class, accuracy = storage_characteristic_matrix(predicted_class, label_class, output_size)

        print('Using Top {} features| accuracy:{:.4f}'.format(K, accuracy))
        all_characteristic_matrix.append(characteristic_matrix)
        ac_score_list.append(accuracy)

    return ac_score_list, all_characteristic_matrix


def run_train(sess, train_X, train_Y, train_step, batch_size):
    X = tf.get_collection('input')[0]
    Y = tf.get_collection('output')[0]

    Iterator = BatchCreate(train_X, train_Y)
    for step in range(1, train_step + 1):
        if step % 100 == 0:
            val_loss, val_accuracy = sess.run(tf.get_collection('validate_ops'), feed_dict={X: train_X, Y: train_Y})
            # val_loss, val_accuracy = sess.run(tf.get_collection('validate_ops'), feed_dict={X: val_X, Y: val_Y})

            print('[%4d] AFS-loss:%.12f AFS-accuracy:%.6f' % (step, val_loss, val_accuracy))
        xs, ys = Iterator.next_batch(batch_size)
        _, A = sess.run(tf.get_collection('train_ops'), feed_dict={X: xs, Y: ys})
    return A


def inference(data_path, train_step, batchsize, f_start, f_end, f_interval):
    train_X, train_Y, test_X, test_Y, class_label = read_mat(data_path)
    args.len = Train_Size = len(train_X)
    total_batch = Train_Size / batchsize
    afs_model.build(total_batch, len(train_X[0]), len(train_Y[0]))
    with tf.Session() as sess:  # 创建上下文
        tf.global_variables_initializer().run()  # 初始化模型参数
        print('== Get feature weight by using AFS ==')
        A = run_train(sess, train_X, train_Y, train_step, batchsize)
    print('==  The Evaluation of AFS ==')
    ac_score_list, characteristic_matrix_summary = run_test(A, train_X, train_Y, test_X,
                                                            test_Y, train_step, len(train_Y[0]), f_start, f_end, f_interval)
    show_feature_selection(ac_score_list, f_start, f_end, f_interval, data_path)
    optimal_result = ac_score_list.index(max(ac_score_list))
    show_confusion_matrix(class_label, characteristic_matrix_summary[optimal_result], data_path)
    print(optimal_result)
    print(max(ac_score_list))


if __name__ == '__main__':
    path = args.data_dir
    args.modelid=1
    args.schedule=0
    inference(path, int(args.max_epochs), int(args.batch_size), 1, 39, 1)
    print("Train Ending")
