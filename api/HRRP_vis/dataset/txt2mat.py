import numpy as np
import scipy.io as sio
import os
import os.path
# 读取index中每一行的路径,并将数据按照标签拼接保存在.mat文件中

indexPath = "H:/WIN_11_DESKTOP/onWorking/207GUI/GUI207_V2.0/api/HRRP_vis/data/valid.txt"
saveRootPath = "H:/WIN_11_DESKTOP/onWorking/207GUI/GUI207_V2.0/api/HRRP_vis/data/test"


# 标签与索引的对应关系
label2index = {
    0: "DQ",
    1: "DT",
    2: "QDZ",
    3: "XQ",
    4: "Z",
}

data_DQ = []
data_DT = []
data_QDZ = []
data_XQ = []
data_Z = []

with open(indexPath, 'r') as f:
    for line in f.readlines():
        sigPath, label = line.strip().split(' ')
        data = np.loadtxt(sigPath)[:, 1]
        eval("data_" + label2index[int(label)]).append(data)


for key in label2index.keys():
    className = label2index[key]
    print(className, np.array(eval("data_" + className)).T.shape)
    # 保存为.mat文件
    matSavePath = saveRootPath+"/"+className
    if not os.path.exists(matSavePath):
        os.makedirs(matSavePath)
    sio.savemat(matSavePath+"/"+className+".mat", {className: np.array(eval("data_" + className)).T})
    
