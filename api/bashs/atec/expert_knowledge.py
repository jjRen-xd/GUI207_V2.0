import numpy as np
from scipy.signal import find_peaks


# 归一化
def max_min_norm(data):
    DATA = []
    for i in range(0, len(data)):
        data_norm = (data[i]-min(data))/(max(data)-min(data))
        DATA.append(data_norm)
    return np.array(DATA)


# 主瓣宽度法
def main_lobe(data):
    f = []  # 从大到小排序幅值
    f_index = []  # 幅值索引
    data_norm = []

    for i in range(0, len(data)):
        i_data = max_min_norm(data[i])  # 归一化处理
        data_norm.append(i_data)
        f.append(np.sort(i_data)[::-1])  # 由大到小排序
        f_index.append(np.argsort(i_data)[::-1])  # 排序索引

    f = np.array(f)
    f_index = np.array(f_index)

    # 最大幅值主瓣宽度附近所有点的幅值置0
    for i in range(0, len(f_index)):
        sc1_amplitude = f[i][0] * 0.5  # 最大幅值
        zb = []  # 主瓣宽度相同点
        l = 0.01
        while (True):
            for j in range(0, len(f[i])):
                if abs(f[i][j] - sc1_amplitude) <= l:
                    zb.append(f_index[i][j])
            if len(zb) != 0:
                break
            else:
                l += 0.01
        fw = []  # 求所有主瓣宽度相同点与散射中心的距离
        for j in range(0, len(zb)):
            dis = abs(zb[j] - f_index[i][0])
            fw.append(dis)
        kd = min(fw)
        for j in range(0, len(f_index[i])):
            if f_index[i][j] <= f_index[i][0] + kd and f_index[i][j] >= f_index[i][0] - kd:
                data_norm[i][f_index[i][j]] = 0

    # 寻找第2个散射中心
    f2 = []
    f2_index = []
    for i in range(0, len(data_norm)):
        i_f = data_norm[i]
        f2.append(np.sort(i_f)[::-1])
        f2_index.append(np.argsort(i_f)[::-1])  # 排序索引

    # 雷达视线上投影长度
    distance = []
    place = np.zeros((len(f_index), 2))
    for i in range(0, len(f_index)):
        distance.append(abs(f_index[i][0] - f2_index[i][0]))

    return distance


# 门限法
def threshold_value_method(data, threshold_value):
    data_norm = []
    for i in range(0, len(data)):
        i_data = max_min_norm(data[i])
        data_norm.append(i_data)
    data_norm = np.array(data_norm)

    distance = []
    for i in range(0, len(data_norm)):
        thre_line = threshold_value
        while(True):
            peaks, _ = find_peaks(data_norm[i], height=thre_line)
            if len(peaks) <= 1:
                thre_line -= 0.01
            else:
                break
        distance.append(abs(peaks[-1]-peaks[0]))

    return distance


# 均值 方差
def mean_var(data):
    mean = []
    variance = []

    for i in range(0, len(data)):
        i_mean = np.mean(data[i])
        i_variance = np.var(data[i])
        mean.append(i_mean)
        variance.append(i_variance)

    mean = np.array(mean)
    variance = np.array(variance)

    return mean, variance


def run_mechanism(data):
    mechanism_knowledge = np.zeros((len(data), 4))
    mechanism_knowledge[:, 0] = main_lobe(data)
    mechanism_knowledge[:, 1] = threshold_value_method(data, 0.6)
    # mechanism_knowledge[:, 2] = threshold_value_method(data, 0.7)
    mechanism_knowledge[:, 2], mechanism_knowledge[:, 3] = mean_var(data)

    return mechanism_knowledge
