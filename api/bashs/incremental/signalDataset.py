# coding=utf-8
import torch
import numpy as np
from torch.utils.data import Dataset
from config import data_path


class PretrainSignalDataset(Dataset):

    def __init__(self, data_type="train"):
        super(PretrainSignalDataset, self).__init__()
        self.data, self.targets = [], []
        self.data = np.load(data_path + data_type + "/old/data.npy", allow_pickle=True)
        self.targets = np.load(data_path + data_type + "/old/label.npy", allow_pickle=True)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class IncrementSignalDataset(Dataset):

    def __init__(self, data_type="train"):
        super(IncrementSignalDataset, self).__init__()
        self.data, self.targets = [], []

        if data_type == "train":
            memory_data = np.load(data_path + data_type + "/memory/data.npy", allow_pickle=True)
            memory_label = np.load(data_path + data_type + "/memory/label.npy", allow_pickle=True)
            current_data = np.load(data_path + data_type + "/current/data.npy", allow_pickle=True)
            current_label = np.load(data_path + data_type + "/current/label.npy",
                                    allow_pickle=True)
            self.data = np.concatenate((memory_data, current_data))
            self.targets = np.concatenate((memory_label, current_label))
        if data_type == "test":
            self.data = np.load(data_path + data_type + "/current/data.npy", allow_pickle=True)
            self.targets = np.load(data_path + data_type + "/current/label.npy",
                                   allow_pickle=True)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    def get_image_class(self, cls):
        return self.data[self.targets == cls], self.targets[self.targets == cls]


class EvalDataset(Dataset):
    def __init__(self, data_type="old"):
        super(EvalDataset, self).__init__()
        self.data, self.targets = [], []
        self.data = np.load(data_path + "test/" + data_type + "/data.npy", allow_pickle=True)
        self.targets = np.load(data_path + "test/" + data_type + "/label.npy", allow_pickle=True)
        # print("EvalDataset:", np.unique(self.targets))
        # print(self.data.shape, self.targets.shape)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_dataset = PretrainSignalDataset(data_type="train")
    test_dataset = PretrainSignalDataset(data_type="test")
