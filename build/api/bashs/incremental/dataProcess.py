# coding=utf-8
import numpy as np
from config import data_path#, raw_data_path, class_name
import os
import scipy.io as sio


def create_dir():

    if not os.path.exists(os.path.join(data_path, "train")):
        os.makedirs(os.path.join(data_path, "train"))
    if not os.path.exists(os.path.join(data_path, "test")):
        os.mkdir(os.path.join(data_path, "test"))

    if not os.path.exists(os.path.join(data_path, "train", "old")):
        os.mkdir(os.path.join(data_path, "train", "old"))
    if not os.path.exists(os.path.join(data_path, "train", "new")):
        os.mkdir(os.path.join(data_path, "train", "new"))
    if not os.path.exists(os.path.join(data_path, "train", "current")):
        os.mkdir(os.path.join(data_path, "train", "current"))
    if not os.path.exists(os.path.join(data_path, "train", "memory")):
        os.mkdir(os.path.join(data_path, "train", "memory"))
    if not os.path.exists(os.path.join(data_path, "test", "old")):
        os.mkdir(os.path.join(data_path, "test", "old"))
    if not os.path.exists(os.path.join(data_path, "test", "new")):
        os.mkdir(os.path.join(data_path, "test", "new"))
    if not os.path.exists(os.path.join(data_path, "test", "current")):
        os.mkdir(os.path.join(data_path, "test", "current"))
    if not os.path.exists(os.path.join(data_path, "test", "observed")):
        os.mkdir(os.path.join(data_path, "test", "observed"))

# "/media/hp/新加卷/data/DD_data/fea/fea_data/"   # (39, 1)
def read_mat_39(raw_data_path):
    data_list = []
    for i in range(1, 7):
        data = sio.loadmat(os.path.join(raw_data_path, "fea"+str(i)+".mat"))["fea1"]
        print(data.shape)
        data_list.append(data)
    
    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []

    for label, data in enumerate(data_list):
        data = np.transpose(data, (1, 0))
        data_len = len(data)
        half_len = int(0.5*data_len)
        train_data = data[0:half_len, :]
        test_data = data[half_len:, :]
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        train_label_list.append(np.repeat(label, len(train_data)))
        test_label_list.append(np.repeat(label, len(test_data)))

    train_data = np.concatenate(train_data_list, 0)
    test_data = np.concatenate(test_data_list, 0)
    train_label = np.concatenate(train_label_list, 0)
    test_label = np.concatenate(test_label_list, 0)
    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

    train_data = train_data[:, np.newaxis, :, np.newaxis]
    test_data = test_data[:, np.newaxis, :, np.newaxis]

    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

    np.save(data_path + "train/data.npy", train_data)
    np.save(data_path + "train/label.npy", train_label)
    np.save(data_path + "test/data.npy", test_data)
    np.save(data_path + "test/label.npy", test_label)

# "/media/hp/新加卷/data/DD_data/tohrrp/hrrp_22june/"下的 hrrp1 - hrrp6.mat
def read_mat_256(raw_data_path):
    data_list = []
    for i in range(1, 7):
        data = sio.loadmat(os.path.join(raw_data_path, "hrrp"+str(i)+".mat"))["hrrp"]
        print(data.shape)
        data_list.append(data)

    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []

    for label, data in enumerate(data_list):
        data = np.transpose(data, (1, 0))
        data_len = len(data)
        half_len = int(0.5*data_len)
        train_data = data[0:half_len, :]
        test_data = data[half_len:, :]
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        train_label_list.append(np.repeat(label, len(train_data)))
        test_label_list.append(np.repeat(label, len(test_data)))

    train_data = np.concatenate(train_data_list, 0)
    test_data = np.concatenate(test_data_list, 0)
    train_label = np.concatenate(train_label_list, 0)
    test_label = np.concatenate(test_label_list, 0)
    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

    train_data = train_data[:, np.newaxis, :, np.newaxis]
    test_data = test_data[:, np.newaxis, :, np.newaxis]

    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

    np.save(data_path + "train/data.npy", train_data)
    np.save(data_path + "train/label.npy", train_label)
    np.save(data_path + "test/data.npy", test_data)
    np.save(data_path + "test/label.npy", test_label)


def read_mat_new(raw_data_path,folder_names):
    g = os.walk(raw_data_path)
    data_list = []
    label_list = []
    label=0
    for className in folder_names:
        tempn=os.path.join(raw_data_path, className)
        wk = os.walk(tempn)
        for path,dir_list,file_list in wk:
            for file_name in file_list:
                if file_name[-4:]==".mat":
                    try:
                        temp = sio.loadmat(os.path.join(path, file_name))[file_name[0:-4]]
                        data_list.append(temp)
                        label_list.append(label)
                        #print(os.path.join(path, file_name))
                        #print("label:"+str(label))
                    except KeyError:
                        print("Warn: read_mat_128_new试图读入"+os.path.join(path, file_name)+",但是其不包含同名变量")
        label+=1

    # for path,dir_list,file_list in g:
    #     for dir_name in dir_list:
    #         if(dir_name=="model_saving"):
    #             continue
    #         tempn=os.path.join(path, dir_name)
    #         print(tempn)
    #         g2 = os.walk(tempn)
    #         for path2,dir_list2,file_list2 in g2:
    #             for file_name in file_list2:
    #                 if file_name[-4:]==".mat":
    #                     try:
    #                         temp = sio.loadmat(os.path.join(path2, file_name))[file_name[0:-4]]
    #                         data_list.append(temp)
    #                         label_list.append(label)
    #                         print(os.path.join(path2, file_name))
    #                         print("label:"+str(label))
    #                     except KeyError:
    #                         print("Warn: read_mat_128_new试图读入"+os.path.join(path2, file_name)+",但是其不包含同名变量")
    #         label+=1

    # print(bigball_hrrp.shape, DT_hrrp.shape, Moxiu_hrrp.shape, smallball_hrrp.shape, taper_hrrp.shape, WD_19_hrrp.shape)
    

    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []

    for i, data in enumerate(data_list):
        data = np.transpose(data, (1, 0))
        data_len = len(data)
        half_len = int(0.5*data_len)
        train_data = data[0:half_len, :]
        test_data = data[half_len:, :]
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        train_label_list.append(np.repeat(label_list[i], len(train_data)))
        test_label_list.append(np.repeat(label_list[i], len(test_data)))

    train_data = np.concatenate(train_data_list, 0)
    test_data = np.concatenate(test_data_list, 0)
    train_label = np.concatenate(train_label_list, 0)
    test_label = np.concatenate(test_label_list, 0)
    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

    train_data = train_data[:, np.newaxis, :, np.newaxis]
    test_data = test_data[:, np.newaxis, :, np.newaxis]

    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

    np.save(data_path + "train/data.npy", train_data)
    np.save(data_path + "train/label.npy", train_label)
    np.save(data_path + "test/data.npy", test_data)
    np.save(data_path + "test/label.npy", test_label)


# "/media/hp/新加卷/data/DD_data/tohrrp/hrrp_data/"  # (128, 1)
# def read_mat_128():
#     bigball1_hrrp = sio.loadmat(os.path.join(raw_data_path, "bigball1_hrrp.mat"))["x_dB"]
#     bigball2_hrrp = sio.loadmat(os.path.join(raw_data_path, "bigball2_hrrp.mat"))["x_dB"]
#     DT_hrrp = sio.loadmat(os.path.join(raw_data_path, "DT_hrrp.mat"))["x_dB"]
#     Moxiu_hrrp = sio.loadmat(os.path.join(raw_data_path, "Moxiu_hrrp.mat"))["x_dB"]
#     smallball1_hrrp = sio.loadmat(os.path.join(raw_data_path, "smallball1_hrrp.mat"))["x_dB"]
#     smallball2_hrrp = sio.loadmat(os.path.join(raw_data_path, "smallball2_hrrp.mat"))["x_dB"]
#     taper1_hrrp = sio.loadmat(os.path.join(raw_data_path, "taper1_hrrp.mat"))["x_dB"]
#     taper2_hrrp = sio.loadmat(os.path.join(raw_data_path, "taper2_hrrp.mat"))["x_dB"]
#     WD_19_hrrp = sio.loadmat(os.path.join(raw_data_path, "WD_19_hrrp.mat"))["x_dB"]
#     bigball_hrrp = np.concatenate((bigball1_hrrp, bigball2_hrrp), axis=1)
#     smallball_hrrp = np.concatenate((smallball1_hrrp, smallball2_hrrp), axis=1)
#     taper_hrrp = np.concatenate((taper1_hrrp, taper2_hrrp), axis=1)
#     # print(bigball_hrrp.shape, DT_hrrp.shape, Moxiu_hrrp.shape, smallball_hrrp.shape, taper_hrrp.shape, WD_19_hrrp.shape)
    
#     data_list = []
#     data_list.append(bigball_hrrp)
#     data_list.append(DT_hrrp)
#     data_list.append(Moxiu_hrrp)
#     data_list.append(smallball_hrrp)
#     data_list.append(taper_hrrp)
#     data_list.append(WD_19_hrrp)

#     train_data_list = []
#     train_label_list = []
#     test_data_list = []
#     test_label_list = []

#     for label, data in enumerate(data_list):
#         data = np.transpose(data, (1, 0))
#         data_len = len(data)
#         half_len = int(0.5*data_len)
#         train_data = data[0:half_len, :]
#         test_data = data[half_len:, :]
#         train_data_list.append(train_data)
#         test_data_list.append(test_data)
#         train_label_list.append(np.repeat(label, len(train_data)))
#         test_label_list.append(np.repeat(label, len(test_data)))

#     train_data = np.concatenate(train_data_list, 0)
#     test_data = np.concatenate(test_data_list, 0)
#     train_label = np.concatenate(train_label_list, 0)
#     test_label = np.concatenate(test_label_list, 0)
#     print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

#     train_data = train_data[:, np.newaxis, :, np.newaxis]
#     test_data = test_data[:, np.newaxis, :, np.newaxis]

#     print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

#     np.save(data_path + "train/data.npy", train_data)
#     np.save(data_path + "train/label.npy", train_label)
#     np.save(data_path + "test/data.npy", test_data)
#     np.save(data_path + "test/label.npy", test_label)



def read_txt(raw_data_path, class_name, snr=2):
    print("read_txt...")
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

    np.save(data_path + "data.npy", data)
    np.save(data_path + "label.npy", label)


# 分割测试集和训练集
def split_test_and_train(test_ratio=0.5, random_seed=1):
    print("split_test_and_train...")
    np.random.seed(random_seed)
    # (738, 512) (738,)
    data = np.load(data_path + "data.npy", allow_pickle=True)
    label = np.load(data_path + "label.npy", allow_pickle=True)

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for i in np.unique(label):
        data_len_ith_class = len(data[label == i])
        indices = np.arange(0, data_len_ith_class)
        np.random.shuffle(indices)
        data_shuffle = data[label == i][indices]
        data_len_ith_class_test = int(data_len_ith_class * test_ratio)
        test_data.append(data_shuffle[:data_len_ith_class_test])
        test_label.append(label[label == i][:data_len_ith_class_test])
        train_data.append(data_shuffle[data_len_ith_class_test:])
        train_label.append(label[label == i][data_len_ith_class_test:])

    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    test_data = np.concatenate(test_data)
    test_label = np.concatenate(test_label)

    # train_data = np.transpose(train_data, (0, 2, 1))
    # test_data = np.transpose(test_data, (0, 2, 1))

    train_data = train_data[:, np.newaxis, :, np.newaxis]

    test_data = test_data[:, np.newaxis, :, np.newaxis]

    np.save(data_path + "train/data.npy", train_data)
    np.save(data_path + "train/label.npy", train_label)
    np.save(data_path + "test/data.npy", test_data)
    np.save(data_path + "test/label.npy", test_label)


def prepare_pretrain_data(old_class=5):
    print("prepare_pretrain_data...")
    train_data = np.load(data_path + "train/data.npy", allow_pickle=True)
    train_label = np.load(data_path + "train/label.npy", allow_pickle=True)
    test_data = np.load(data_path + "test/data.npy", allow_pickle=True)
    test_label = np.load(data_path + "test/label.npy", allow_pickle=True)
    old_train_data = []
    old_train_label = []
    old_test_data = []
    old_test_label = []
    new_train_data = []
    new_train_label = []
    new_test_data = []
    new_test_label = []
    for i in range(old_class):
        old_train_data.append(train_data[train_label == i])
        old_train_label.append(train_label[train_label == i])
        old_test_data.append(test_data[test_label == i])
        old_test_label.append(test_label[test_label == i])
    for i in range(old_class, 11):
        new_train_data.append(train_data[train_label == i])
        new_train_label.append(train_label[train_label == i])
        new_test_data.append(test_data[test_label == i])
        new_test_label.append(test_label[test_label == i])
    old_train_data = np.concatenate(old_train_data)
    old_train_label = np.concatenate(old_train_label)
    old_test_data = np.concatenate(old_test_data)
    old_test_label = np.concatenate(old_test_label)
    new_train_data = np.concatenate(new_train_data)
    new_train_label = np.concatenate(new_train_label)
    new_test_data = np.concatenate(new_test_data)
    new_test_label = np.concatenate(new_test_label)
    np.save(data_path + "train/old/data.npy", old_train_data)
    np.save(data_path + "train/old/label.npy", old_train_label)
    np.save(data_path + "test/old/data.npy", old_test_data)
    np.save(data_path + "test/old/label.npy", old_test_label)
    np.save(data_path + "train/new/data.npy", new_train_data)
    np.save(data_path + "train/new/label.npy", new_train_label)
    np.save(data_path + "test/new/data.npy", new_test_data)
    np.save(data_path + "test/new/label.npy", new_test_label)
    np.save(data_path + "test/observed/data.npy", old_test_data)
    np.save(data_path + "test/observed/label.npy", old_test_label)


def prepare_increment_data(task_class):
    print("prepare_increment_data...")
    observed_test_data = np.load(data_path + "test/observed/data.npy")
    observed_test_label = np.load(data_path + "test/observed/label.npy")
    new_train_data = np.load(data_path + "train/new/data.npy")
    new_train_label = np.load(data_path + "train/new/label.npy")
    new_test_data = np.load(data_path + "test/new/data.npy")
    new_test_label = np.load(data_path + "test/new/label.npy")
    current_task_data_train = []
    current_task_label_train = []
    current_task_data_test = []
    current_task_label_test = []
    for cls in task_class:
        current_task_data_train.append(new_train_data[new_train_label == cls])
        current_task_label_train.append(new_train_label[new_train_label == cls])
        current_task_data_test.append(new_test_data[new_test_label == cls])
        current_task_label_test.append(new_test_label[new_test_label == cls])
    current_task_data_train = np.concatenate(current_task_data_train)
    current_task_label_train = np.concatenate(current_task_label_train)
    current_task_data_test = np.concatenate(current_task_data_test)
    current_task_label_test = np.concatenate(current_task_label_test)
    observed_test_data = np.concatenate((current_task_data_test, observed_test_data))
    observed_test_label = np.concatenate((current_task_label_test, observed_test_label))
    np.save(data_path + "train/current/data.npy", current_task_data_train)
    np.save(data_path + "train/current/label.npy", current_task_label_train)
    np.save(data_path + "test/current/data.npy", current_task_data_test)
    np.save(data_path + "test/current/label.npy", current_task_label_test)
    np.save(data_path + "test/observed/data.npy", observed_test_data)
    np.save(data_path + "test/observed/label.npy", observed_test_label)


# 新类训练样本保留reduce_sample%
def prepare_increment_data_reduce(task_class,reduce_sample):
    print("prepare_increment_data...")
    observed_test_data = np.load(data_path + "test/observed/data.npy")
    observed_test_label = np.load(data_path + "test/observed/label.npy")
    new_train_data = np.load(data_path + "train/new/data.npy")
    new_train_label = np.load(data_path + "train/new/label.npy")
    new_test_data = np.load(data_path + "test/new/data.npy")
    new_test_label = np.load(data_path + "test/new/label.npy")
    current_task_data_train = []
    current_task_label_train = []
    current_task_data_test = []
    current_task_label_test = []
    for cls in task_class:
        data_tmp = new_train_data[new_train_label == cls]
        reduce_len = int(len(data_tmp)*reduce_sample)
        current_task_data_train.append(data_tmp[:reduce_len, ...])
        current_task_label_train.append(new_train_label[new_train_label == cls][:reduce_len, ...])
        current_task_data_test.append(new_test_data[new_test_label == cls])
        current_task_label_test.append(new_test_label[new_test_label == cls])

    current_task_data_train = np.concatenate(current_task_data_train)
    current_task_label_train = np.concatenate(current_task_label_train)
    current_task_data_test = np.concatenate(current_task_data_test)
    current_task_label_test = np.concatenate(current_task_label_test)
    observed_test_data = np.concatenate((current_task_data_test, observed_test_data))
    observed_test_label = np.concatenate((current_task_label_test, observed_test_label))
    np.save(data_path + "train/current/data.npy", current_task_data_train)
    np.save(data_path + "train/current/label.npy", current_task_label_train)
    np.save(data_path + "test/current/data.npy", current_task_data_test)
    np.save(data_path + "test/current/label.npy", current_task_label_test)
    np.save(data_path + "test/observed/data.npy", observed_test_data)
    np.save(data_path + "test/observed/label.npy", observed_test_label)


def get_number_of_each_class():
    train_data = np.load("./train/data.npy", allow_pickle=True)
    print(train_data.shape)
    train_label = np.load(data_path + "train/label.npy", allow_pickle=True)
    test_data = np.load(data_path + "test/data.npy", allow_pickle=True)
    test_label = np.load(data_path + "test/label.npy", allow_pickle=True)
    data = np.load(data_path + "data.npy", allow_pickle=True)
    label = np.load(data_path + "label.npy", allow_pickle=True)
    data = np.concatenate(data)
    label = np.concatenate(label)
    train_num_list = []
    test_num_list = []
    num_list = []
    for i in range(11):
        train_label_i = train_label[train_label == i]
        test_label_i = test_label[test_label == i]
        label_i = label[label == i]
        train_num_list.append(len(train_label_i))
        test_num_list.append(len(test_label_i))
        num_list.append(len(label_i))
    sum_test = 0
    sum_train = 0
    for i in test_num_list:
        sum_test = sum_test + i
    for i in train_num_list:
        sum_train = sum_train + i
    print("each_class_of_train_data:", train_num_list, sum_train)
    print("each_class_of_test_data:", test_num_list, sum_test)
    print("each_class_of_all_data:", num_list)


# if __name__ == '__main__':
    # read_txt(snr=2)
    # split_test_and_train(test_ratio=0.5, random_seed=1)
    # prepare_pretrain_data(old_class=5)
    # prepare_increment_data([4])
    
    # create_dir()

    # read_mat()
