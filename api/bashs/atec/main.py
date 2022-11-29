import os,shutil
import sys
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from net_fea import net_fea_extract
from mapping_net import fea_mapping
from data_process import trans_norm
from expert_knowledge import run_mechanism
from sklearn.metrics import classification_report, confusion_matrix
from data_process import norm_one, show_confusion_matrix


parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data',default="D:/lyh/GUI207_V2.0/db/datasets/hrrp_1x128")
parser.add_argument('--work_dir', help='the directory of the training data',default="../db/trainLogs")
parser.add_argument('--time', help='the directory of the training data',default="2022-09-21-21-52-17")
parser.add_argument('--model_name', help='the directory of the training data',default="model")
parser.add_argument('--batch_size', help='the number of batch size',default=32)
parser.add_argument('--max_epochs', help='the number of epochs',default=10)
parser.add_argument('--new_data_dir', help='the feature blending data',default="../db/datasets/tempFeatureDataFromHRRP")
parser.add_argument('--modeldir', help="model saved path", default="../db/models")

args = parser.parse_args()


# 数据归一化
def data_normalization(data):
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


# 从.mat文件读取数据并预处理
def read_mat(read_path):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    args.classNum=len(file_name)
    class_data_num = []
    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的.mat文件名称
        class_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[0]  # 类别的.mat文件路径
        matrix_base = os.path.basename(class_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        class_data = sio.loadmat(class_path)[matrix_name].T  # 读入.mat文件，并转置

        # 设置标签
        label = np.zeros((len(class_data), len(folder_name)))
        label[:, i] = 1

        # 统计每个类别中样本数目
        class_data_num.append(len(class_data))

        # 划分训练数据集和测试数据集
        x_train = class_data[:int(len(class_data)/2), :]
        x_test = class_data[int(len(class_data)/2):, :]
        y_train = label[:int(len(class_data)/2), :]
        y_test = label[int(len(class_data)/2):, :]

        if i == 0:
            train_x = x_train
            test_x = x_test
            train_y = y_train
            test_y = y_test
        else:
            train_X = np.concatenate((train_x, x_train))
            train_Y = np.concatenate((train_y, y_train))
            test_X = np.concatenate((test_x, x_test))
            test_Y = np.concatenate((test_y, y_test))

            train_x = train_X
            train_y = train_Y
            test_x = test_X
            test_y = test_Y

    return train_X, train_Y, test_X, test_Y, len(folder_name), folder_name, class_data_num


# 训练过程中准确率曲线
def train_acc(epoch, acc):
    x = np.arange(epoch+1)[1:]
    plt.figure()
    plt.plot(x, acc)
    plt.scatter(x, acc)
    plt.grid()
    plt.title('Training accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.work_dir+'./training_accuracy.jpg', dpi=300)


# 验证准确率曲线
def val_acc(v_acc):
    x = np.arange(len(v_acc)+1)[1:]
    plt.figure()
    plt.plot(x, v_acc)
    plt.scatter(x, v_acc)
    plt.grid()
    plt.title('Verification accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Times', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.work_dir+'./verification_accuracy.jpg', dpi=300)

# 特征交融数据存储
def fea_mapping_save(train_fea, test_fea, folder_name, class_data_num):
    the_class_num_train = 0
    previous_class_num_train = 0
    the_class_num_test = 0
    previous_class_num_test = 0
    for i in range(0, len(folder_name)):
        the_class_num_train += int(class_data_num[i]/2)
        the_class_num_test += class_data_num[i] - int(class_data_num[i] / 2)
        fea_mapping_class = np.zeros((class_data_num[i], train_fea.shape[1]))
        fea_mapping_class[:int(class_data_num[i]/2), :] = train_fea[previous_class_num_train:the_class_num_train, :]
        fea_mapping_class[int(class_data_num[i]/2):, :] = test_fea[previous_class_num_test:the_class_num_test, :]
        previous_class_num_train = the_class_num_train
        previous_class_num_test = the_class_num_test
        if os.path.exists(args.new_data_dir+'/'+folder_name[i]):
            sio.savemat(args.new_data_dir+'/'+folder_name[i]+'/'+folder_name[i]+'.mat', mdict={folder_name[i]: fea_mapping_class.T})
        else:
            os.makedirs(args.new_data_dir+'/'+folder_name[i])
            sio.savemat(args.new_data_dir+'/'+folder_name[i]+'/'+folder_name[i]+'.mat', mdict={folder_name[i]: fea_mapping_class.T})


# 学习模块
def rcn_model(train_x, train_y, test_x, test_y, epoch, batch_size):
    rcn_model = keras.Sequential([
        keras.layers.Conv1D(16, kernel_size=1, padding='valid', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(train_y[0]), activation='softmax')
    ])
    rcn_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    save_model_path = args.work_dir + '/model/fea_ada_trans.hdf5'
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='lr', factor=0.99, patience=3,
                                                             verbose=0, min_lr=0.0001)
    checkpoint = keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learn_rate_reduction]
    h = rcn_model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, shuffle=True, validation_data=(test_x, test_y),
                  callbacks=callbacks_list, verbose=1, validation_freq=10)
    h_parameter = h.history
    train_acc(int(args.max_epochs), h_parameter['accuracy'])
    val_acc(h_parameter['val_accuracy'])
    test_model = keras.models.load_model(save_model_path)
    Y_test = np.argmax(test_y, axis=1)
    Y_pred = test_model.predict_classes(test_x)
    args.valAcc=round(max(h_parameter['val_accuracy'])*100,2)
    return Y_test, Y_pred


# 特征交融
def run_mapping(train_x, train_y, test_x, test_y, epoch, batch_size):
    train_Y, test_Y = run_mechanism(train_x), run_mechanism(test_x)
    train_x, test_x = trans_norm(train_x), trans_norm(test_x)
    train_x = np.expand_dims(train_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)

    # 神经网络特征提取
    path_one = args.work_dir + '/model/net_fea.hdf5'
    train_fea, test_fea = net_fea_extract(path_one, train_x, train_y, test_x, test_y, epoch, batch_size)

    # 特征适应变换
    path_two = args.work_dir + '/model/fit_model.hdf5'
    train_mapping, test_mapping = fea_mapping(path_two, train_x, train_Y, test_x, test_Y, epoch, batch_size)

    train_new = np.zeros((len(train_x), len(train_fea[0])+len(train_mapping[0])))
    test_new = np.zeros((len(test_x), len(test_fea[0])+len(test_mapping[0])))

    train_new[:, :len(train_mapping[0])] = norm_one(train_fea, train_mapping)
    test_new[:, :len(test_mapping[0])] = norm_one(test_fea, test_mapping)
    train_new[:, len(train_mapping[0]):len(train_fea[0])+len(train_mapping[0])] = train_fea
    test_new[:, len(test_mapping[0]):len(test_fea[0])+len(test_mapping[0])] = test_fea

    return train_new, test_new


def inference(train_x, train_y, test_x, test_y, folder_name, class_data_num):
    batch_size = int(args.batch_size)
    epoch = int(args.max_epochs)
    train_new, test_new = run_mapping(train_x, train_y, test_x, test_y, epoch, batch_size)
    fea_mapping_save(train_new, test_new, folder_name, class_data_num)
    len_train = len(train_new)
    len_test = len(test_new)
    train_shuffle = np.arange(len_train)
    test_shuffle = np.arange(len_test)
    np.random.shuffle(train_shuffle)
    np.random.shuffle(test_shuffle)
    new_train = train_new[train_shuffle, :]
    train_y = train_y[train_shuffle, :]
    new_test = test_new[test_shuffle, :]
    test_y = test_y[test_shuffle, :]

    new_train = np.expand_dims(new_train, axis=-1)
    new_test = np.expand_dims(new_test, axis=-1)
    Y_test, Y_pred = rcn_model(new_train, train_y, new_test, test_y, epoch, batch_size)
    characteristic_matrix = confusion_matrix(Y_test, Y_pred)
    class_label = folder_name  # 标签显示
    show_confusion_matrix(args.work_dir, class_label, characteristic_matrix)
    print(classification_report(Y_test, Y_pred, digits=4))

def generator_model_documents(args):
    from xml.dom.minidom import Document
    doc = Document()  #创建DOM文档对象
    root = doc.createElement('ModelInfo') #创建根元素
    doc.appendChild(root)
    
    model_type = doc.createElement('FEA_RELE')
    #model_type.setAttribute('typeID','1')
    root.appendChild(model_type)

    model_item = doc.createElement(args.model_name+'.trt')
    #model_item.setAttribute('nameID','1')
    model_type.appendChild(model_item)

    model_infos = {
        'name':str(args.model_name),
        'type':'FEA_RELE',
        'algorithm':'ATEC',
        'framework':'keras',
        'accuracy':str(args.valAcc),
        'trainDataset':args.data_dir.split("/")[-1],
        'trainEpoch':str(args.max_epochs),
        'trainLR':'0.001',
        'class':str(args.classNum), 
        'PATH':os.path.abspath(os.path.join(args.modeldir,args.model_name+'.trt')),
        'batch':str(args.batch_size),
        'note':'-'
    } 

    for key in model_infos.keys():
        info_item = doc.createElement(key)
        info_text = doc.createTextNode(model_infos[key]) #元素内容写入
        info_item.appendChild(info_text)
        model_item.appendChild(info_item)

    with open(os.path.join(args.modeldir,args.model_name+'.xml'),'w') as f:
        doc.writexml(f,indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')

    shutil.copy(args.work_dir+"/"+args.model_name+".trt",os.path.join(args.modeldir,args.model_name+'.trt'))
    shutil.copy(args.work_dir+"/model/fea_ada_trans.hdf5",os.path.join(args.modeldir,args.model_name+'.hdf5'))
    shutil.copy(args.work_dir+"/"+"confusion_matrix.jpg",os.path.join(args.modeldir,'confusion_matrix.jpg'))
    shutil.copy(args.work_dir+"/"+"training_accuracy.jpg",os.path.join(args.modeldir,'training_accuracy.jpg'))
    shutil.copy(args.work_dir+"/"+"verification_accuracy.jpg",os.path.join(args.modeldir,'verification_accuracy.jpg'))

if __name__ == '__main__':
    datasetName = args.data_dir.split("/")[-1]
    #datasetName = args.new_data_dir.split("/")[-1]
    args.work_dir = args.work_dir+"/"+args.time+'-'+datasetName+'-'+args.model_name
    train_x, train_y, test_x, test_y, class_num, folder_name, class_data_num = read_mat(args.data_dir)
    inference(train_x, train_y, test_x, test_y, folder_name, class_data_num)
    print("start model convert")
    sys.stdout.flush()
    cmd="python ../api/bashs/hdf52trt.py --model_type ATEC --work_dir "+args.work_dir+" --model_name "+args.model_name
    os.system(cmd)
    args.modeldir = args.modeldir+'/'+args.model_name
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)
    generator_model_documents(args)
    print("Train Ended:")