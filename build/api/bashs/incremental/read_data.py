import numpy as np
import os
from config import class_name, raw_data_path, npy_data_path


def read_txt(snr=2):
    data_dir = os.path.join(raw_data_path, str(snr))

    data_list = []
    label_list = []
    for index, cls in enumerate(class_name):
        cls_path = os.path.join(data_dir, cls)
        all_files = os.listdir(cls_path)
        cls_datas = []
        cls_labels = []
        for file_name in all_files:
            file_path = os.path.join(cls_path, file_name)
            file = open(file_path, "r")
            data_lines = file.readlines()
            datas = []
            for data in data_lines:
                splits = data.split("\t")
                splits = splits[1].split("\n")
                data = float(splits[0])
                datas.append(data)
            cls_datas.append(datas)
            cls_labels.append(index)
        data_list.append(cls_datas)
        label_list.append(cls_labels)

    data = np.array(data_list)
    label = np.array(label_list)

    data = np.concatenate(data, 0)
    label = np.concatenate(label, 0)

    np.save(npy_data_path + "data_{}.npy".format(snr), data)
    np.save(npy_data_path + "label_{}.npy".format(snr), label)
    return data, label


def split_test_and_train(test_percent=0.5, snr=2, random_seed=1):
    np.random.seed(random_seed)

    data = np.load(npy_data_path + "data_{}.npy".format(snr))
    label = np.load(npy_data_path + "label_{}.npy".format(snr))

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for i in range(len(data)):
        data_len_ith_class = len(data[i])
        indices = np.arange(0, data_len_ith_class)
        np.random.shuffle(indices)
        data_shuffle = data[i][indices]
        data_len_ith_class_test = int(data_len_ith_class * test_percent)
        test_data.append(data_shuffle[:data_len_ith_class_test])
        test_label.append(label[i][:data_len_ith_class_test])
        train_data.append(data_shuffle[data_len_ith_class_test:])
        train_label.append(label[i][data_len_ith_class_test:])

    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    test_data = np.concatenate(test_data)
    test_label = np.concatenate(test_label)

    train_data = np.transpose(train_data, (0, 2, 1))
    test_data = np.transpose(test_data, (0, 2, 1))

    train_data = train_data[:, np.newaxis, :, :]
    test_data = test_data[:, np.newaxis, :, :]

    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    # data, label = read_txt(snr=2)
    split_test_and_train(test_percent=0.5, snr=2, random_seed=1)
