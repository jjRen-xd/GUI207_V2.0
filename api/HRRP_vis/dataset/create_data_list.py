import sys
sys.path.append('../')

import os
import random
from tqdm import tqdm


if not os.path.exists('./data'):
    os.makedirs('./data')

train_txt = open('./data/train.txt', 'w')
val_txt = open('./data/valid.txt', 'w')
label_txt = open('./data/label_list.txt', 'w')

label_list = []
data_root = './dataset/HRRP_20220508/'
for dir in tqdm(os.listdir(data_root)):
    if dir not in label_list:
        label_list.append(dir)
        label_txt.write('{} {}\n'.format(dir, str(len(label_list)-1)))
        data_path = os.path.join(data_root, dir)
        train_list = random.sample(os.listdir(data_path), 
                                   int(len(os.listdir(data_path))*0.8))
        for im in train_list:
            train_txt.write('{}/{}/{} {}\n'.format(data_root, dir, im, str(len(label_list)-1)))
        for im in os.listdir(data_path):
            if im in train_list:
                continue
            else:
                val_txt.write('{}/{}/{} {}\n'.format(data_root, dir, im, str(len(label_list)-1)))



