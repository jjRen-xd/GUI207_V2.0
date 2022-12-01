import os
import gc
import shutil
import argparse

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

import numpy as np
import scipy.io as scio
import tensorflow as tf
from tensorflow.keras.models import Model
from contextlib import redirect_stdout   
from tensorflow.keras.utils import plot_model  


name2label = {
    "Big_ball": 0,
    "Cone": 1,
    "Cone_cylinder": 2,
    "DT": 3,
    "Small_ball": 4,
    "Spherical_cone": 5,
}


def saveModelInfo(model, modelPath):
    rootPath = os.path.dirname(modelPath)
    modelName = os.path.basename(modelPath).split('.')[0]

    # 保存模型所有基本信息
    with open(rootPath + '/'+ modelName + "_modelInfo.txt", 'w') as f:
        with redirect_stdout(f):
            model.summary(line_length=200, positions=[0.30,0.60,0.7,1.0])
        
    # 保存模型所有层的名称至xml文件
    from xml.dom.minidom import Document
    xmlDoc = Document()
    child_1 = xmlDoc.createElement(modelName)
    xmlDoc.appendChild(child_1)
    child_2 = xmlDoc.createElement(modelName)
    child_1.appendChild(child_2)
    for layer in model.layers:  
        layer = layer.name.replace("/", "_")
        nodeList = layer.split("_")
        for i in range(len(nodeList)):
            modeName = nodeList[i].strip()
            if modeName.isdigit():
                modeName = "_" + modeName
            if i == 0:
                # 如果以modeName为名的节点已经存在，就不再创建，直接挂
                if len(child_2.getElementsByTagName(modeName)) == 0:
                    node1 = xmlDoc.createElement(modeName)
                    child_2.appendChild(node1)
                else:
                    node1 = child_2.getElementsByTagName(modeName)[0]
            elif i == 1:
                if len(node1.getElementsByTagName(modeName)) == 0:
                    node2 = xmlDoc.createElement(modeName)
                    node1.appendChild(node2)
                else:
                    node2 = node1.getElementsByTagName(modeName)[0]
            elif i == 2:
                if len(node2.getElementsByTagName(modeName)) == 0:
                    node3 = xmlDoc.createElement(modeName)
                    node2.appendChild(node3)
                else:
                    node3 = node2.getElementsByTagName(modeName)[0]
            elif i == 3:
                if len(node3.getElementsByTagName(modeName)) == 0:
                    node4 = xmlDoc.createElement(modeName)
                    node3.appendChild(node4)
                else:
                    node4 = node3.getElementsByTagName(modeName)[0]
    f = open(rootPath + '/'+ modelName + "_struct.xml", "w")
    xmlDoc.writexml(f, addindent='\t', newl='\n', encoding="utf-8", standalone="yes")
    f.close()

    # 保存模型结构图
    if not os.path.exists(rootPath + '/'+ modelName + "_structImage"):
        os.makedirs(rootPath + '/'+ modelName + "_structImage")
    plot_model(model, to_file = rootPath + '/'+ modelName + "_structImage/framework.png", show_shapes=True, show_layer_names=True)


def read_mat(matPath, repeat=True):
    ''' 读取.mat文件 '''
    mat = scio.loadmat(matPath)
    matrix_base = os.path.basename(matPath)
    labelName = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
    signals = scio.loadmat(matPath)[labelName].T  # 读入.mat文件,并转置
    signals = data_normalization(signals)  # 归一化处理
    labels = [name2label[labelName]] * len(signals)  # 标签   

    if repeat:
        # 数据复制64次
        class_data_picture = []
        for j in range(0, len(signals)):
            class_data_one = signals[j]
            empty = np.zeros((len(class_data_one), 64))
            for k in range(0, len(class_data_one)):
                empty[k, :] = class_data_one[k]
            class_data_picture.append(empty)
        class_data_picture = np.array(class_data_picture)

        return class_data_picture, labels, labelName
    else:
        return signals, labels, labelName

def data_normalization(data):
    """
        Func:
            数据归一化
        Args:
            data: 待归一化的数据
        Return:
            data: 归一化后的数据
    """
    for i in range(0, len(data)):
        data[i] -= np.min(data[i])
        data[i] /= np.max(data[i])
    return data


def capActAndGrad(signal, label, checkpoint_path, targetLayerName='conv5_block16_2_conv', top1 = False, saveInfo = False):

    
    # 模型构建
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.applications.densenet.DenseNet121(include_top=True, weights=None,
    #                                                    input_tensor=None, input_shape=(128, 64, 1), pooling=None, classes=6))
    # model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    # learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='lr', patience=3, verbose=1, factor=0.99, min_lr=0.00001)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint('./densenet121_hrrp_128.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint, learning_rate_reduction]
    # model.summary()

    # Load model
    model = tf.keras.models.load_model(checkpoint_path)
    prediction = model.predict(signal[None, ..., None])
    prediction_idx = np.argmax(prediction)

    # Target hidden layer
    if ("baseline" in checkpoint_path) or ("ATEC" in checkpoint_path):   # baseline模型是在一级层下
        modelInner = model
    else:                               # 其他模型是经过了sequential,在二级层下
        modelInner = model.get_layer(model.layers[0].name)
    if saveInfo:
        saveModelInfo(modelInner, checkpoint_path)
    target_layer = modelInner.get_layer(targetLayerName)
    gradient_model = Model([modelInner.inputs], [target_layer.output, modelInner.output])

    # Compute Gradient of Top Predicted Class
    with tf.GradientTape() as tape:
        activations, prediction = gradient_model(signal[None, ...])
        if top1:
            score = prediction[:, prediction_idx]
        else:
            score = prediction[:, label]

    # Gradient() computes the gradient using operations recorded in context of this tape
    gradients = tape.gradient(score, activations)

    # Change the position of channel axes, for visualization
    if activations.ndim == 2:
        activations = activations[:, :, None, None]
        gradients = gradients[:, :, None, None]
    elif "baseline" in checkpoint_path or activations.ndim == 3:
        activations = tf.transpose(activations, perm=[0, 2, 1])[:, :, :, None]
        gradients = tf.transpose(gradients, perm=[0, 2, 1])[:, :, :, None]
    elif activations.ndim == 4:
        activations = tf.transpose(activations, perm=[0, 3, 1, 2])
        gradients = tf.transpose(gradients, perm=[0, 3, 1, 2])

    return activations.numpy(), gradients.numpy(), prediction_idx


def shuffle(data, label):
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize a keras model'
    )
    parser.add_argument(
        '--checkpoint', 
        # default="../../db/models/HRRP_128_densenet121_c6_keras/HRRP_128_densenet121_c6_keras.hdf5",
        default="H:\WIN_11_DESKTOP\onWorking/207GUI\GUI207_V2.0\db\models\HRRP_128_baselineCNN_c6_keras\HRRP_128_baselineCNN_c6_keras.hdf5",
        type=str,
        help='checkpoint file'
    )
    parser.add_argument(
        '--visualize_layer',
        # default="conv5_block16_2_conv",
        default="dense_2",
        type=str,
        help='Name of the hidden layer of the model to visualize'
    )
    parser.add_argument(
        '--mat_path',
        default="H:\WIN_11_DESKTOP\onWorking/207GUI\GUI207_V2.0\db\datasets\HRRP_simulate_128xN_c6\Big_ball\Big_ball.mat",
        type=str,
        help='The .mat path of signal to visualize'
    )
    parser.add_argument(
        '--mat_idx',
        default=0,
        type=int,
        help='The .mat index of visual signal'
    )
    parser.add_argument(
        '--act_or_grad',
        default=1,
        type=int,
        help='Visual features or gradients'
    )
    parser.add_argument(
        '--save_path',
        default="./figs/fea_output",
        type=str,
        help='The path of feature map to save'
    )
    parser.add_argument(
        '--save_model_info',
        default=0,
        type=bool,
    )
    args = parser.parse_args()

    # 读取数据
    repeatData = False if ("baseline" in args.checkpoint or "ATEC" in args.checkpoint) else True
    signals, labels, labelName = read_mat(args.mat_path, repeatData)
    # (275, 128, 64), (275, 6), (275, 128, 64), (275, 6)
    # 0:'Cone', 1:'Cone_cylinder', 2:'DT', 3:'WD', 4:'bigball', 5:'smallball'
    signal = signals[args.mat_idx]
    label = labels[args.mat_idx]    # one-hot -> index; (275, 6) -> (275,)
    
    # 捕获激活和梯度
    activations, gradients, prediction_idx = capActAndGrad(
        signal, label, 
        args.checkpoint, 
        args.visualize_layer,
        top1=True,          # True: top1, False: Ground Truth
        saveInfo=args.save_model_info
    )

    # 检查保存路径
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 保存激活图/梯度图
    if args.act_or_grad == 0:
        visFeatures = np.mean(activations[0], axis=2)
    elif args.act_or_grad == 1:
        visFeatures = np.mean(gradients[0], axis=2)
        # for i in range(len(visFeatures)):
        #     visFeatures[i] -= np.min(visFeatures[i])
        #     visFeatures[i] /= np.max(visFeatures[i])
        if np.max(np.abs(visFeatures)) != 0:    # 规整到一个合适的范围
            visFeatures *= 1/np.max(np.abs(visFeatures))
    else:
        raise RuntimeError('args.act_or_grad must be 0 or 1')

    if visFeatures.shape[1] == 1:
        plt.figure(figsize=(18, 4), dpi=100)
        plt.grid(axis="y")
        plt.bar(range(len(visFeatures[:, 0])), visFeatures[:, 0])
        plt.title("Signal Type: "+labelName+"    Model Layer: "+"model."+args.visualize_layer)
        plt.xlabel("Number of units")
        plt.ylabel("Activation value")
        plt.savefig(args.save_path + "/FC_vis.png")

        plt.clf()
        plt.close()
        gc.collect()
    else:
        for idx, featureMap in enumerate(visFeatures):   # for every channels
            plt.figure(figsize=(18, 4))
            plt.title("Signal Type: "+labelName+"    Model Layer: "+"model."+args.visualize_layer)
            plt.xlabel('N')
            plt.grid(axis="y")
            plt.ylabel("Value")
            plt.plot(featureMap, linewidth=2, label = 'Hidden layer features')
            plt.legend(loc="upper right")
            plt.savefig(args.save_path + "/" + str(idx+1) + ".png")
            # plt.show()
            plt.clf()
            plt.close()
            gc.collect()

    print("finished")






