import os
import scipy.io as sio
import argparse
import numpy as np
from scipy import fftpack
from scipy.signal import stft
import pywt


parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument('--data_path', help='the directory of the data')
parser.add_argument('--save_path', help='the path to save processed data')


args = parser.parse_args()


# 数据归一化
def fea_normalization(data):
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


# 特征提取
def feature_extraction(read_path, save_path):

    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])

    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path+'/'+folder_name[i])  # 获取类别文件夹下的.mat文件名称
        class_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[0]  # 类别的.mat文件路径
        matrix_base = os.path.basename(class_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        class_data = sio.loadmat(class_path)[matrix_name].T  # 读入.mat文件，并转置

        new_class_data = []  # 特征提取后的数据

        # 各种特征提取方式处理数据
        for j in range(0, len(class_data)):
            class_data_process = []
            feature_extraction_num_process = []

            fea_1 = pywt.dwt(class_data[j], 'haar')  # 小波变换，haar
            class_data_process.extend(fea_1[0])
            feature_extraction_num_process.append(len(fea_1[0]))

            fea_2 = pywt.dwt(class_data[j], 'sym3')  # 小波变换，sym3
            class_data_process.extend(fea_2[0])
            feature_extraction_num_process.append(len(fea_2[0]))

            fea_3 = pywt.dwt(class_data[j], 'db2')  # 小波变换，db2
            class_data_process.extend(fea_3[0])
            feature_extraction_num_process.append(len(fea_3[0]))

            fea_4 = pywt.dwt(class_data[j], 'coif1')  # 小波变换，coif1
            class_data_process.extend(fea_4[0])
            feature_extraction_num_process.append(len(fea_4[0]))

            fea_5 = fftpack.hilbert(class_data[j])  # 希尔伯特变换
            class_data_process.extend(fea_5)
            feature_extraction_num_process.append(len(fea_5))

            fea_6 = fftpack.fft(class_data[j])  # 快速傅里叶变换
            class_data_process.extend((abs(fea_6[1:])))
            feature_extraction_num_process.append(len(abs(fea_6[1:])))

            fea_7 = stft(class_data[j])  # 短时傅里叶变换
            ds_stft = []
            for k in range(0, len(abs(fea_7[2]))):
                ds_stft.extend(abs(fea_7[2])[k])
            class_data_process.extend(ds_stft)
            feature_extraction_num_process.append(len(ds_stft))

            new_class_data.append(class_data_process)

        new_class_data_norm = fea_normalization(new_class_data)  # 归一化
        new_class_data_norm = np.array(new_class_data_norm).transpose()  # 将列表转换为数组

        if not os.path.exists(save_path + '/' + folder_name[i]):  # 判断文件夹是否存在
            os.makedirs(save_path + '/' + folder_name[i])  # 建立转换后存储类别数据的文件夹
        new_matrix_name = 'feature'
        mat_path = save_path + '/' + folder_name[i] + '/' + new_matrix_name + os.path.splitext(matrix_base)[-1]  # 转换完成后的.mat文件存储路径
        sio.savemat(mat_path, {new_matrix_name: new_class_data_norm})  # 保存到.mat文件


if __name__ == '__main__':
    feature_extraction(args.data_path, args.save_path)
