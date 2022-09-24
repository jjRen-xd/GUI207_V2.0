# -*- coding: utf-8 -*-    #
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw_confusion_matrix(classesf, confusion_matrix):
    classes = classesf.split("#")
    plt.figure()
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []  #�ٷֱ�(�б���)
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(�еĳ��ȣ��еĳ���)
    pshow = np.array(pshow).reshape(length, length)
    config = {"font.family": 'Times New Roman'}  # ������������
    matplotlib.rcParams.update(config)
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
    #plt.tight_layout()
    plt.savefig('D:/confusion_matrix.jpg', dpi=300)

# data1 = np.full((6,6),2, dtype = int )
# draw_confusion_matrix("A#B#C#D#E#F#",data1)