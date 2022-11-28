import os,re,shutil
import argparse
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


window_len, step ,time_of_copy= 128, 1, 64

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data')
parser.add_argument('--time', help='the directory of the training data',default="2022-09-21-21-52-17")
parser.add_argument('--work_dir', help='the directory of the trainingLogs',default="../db/trainLogs")
parser.add_argument('--model_name', help='the Name of the model',default="model")
parser.add_argument('--batch_size', help='the number of batch size',default=32)
parser.add_argument('--max_epochs', help='the number of epochs',default=1)
parser.add_argument('--modeldir', help="model saved path", default="../db/models")
parser.add_argument('--class_number', help="class_number", default="6")
args = parser.parse_args()


# 归一化
def trans_norm(data):
    data_trans = list(map(list, zip(*data)))
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(np.array(data_trans))
    trans_data = np.array(list(map(list, zip(*data_norm))))

    return trans_data


# 滑动窗口取值
def sliding_window(data):
    DATA = []
    data_len = len(data)
    win_start = 0
    win_end = window_len
    for i in range(data_len):
        win_data = data[win_start:win_end]
        DATA.append(win_data)
        win_start += step
        win_end += step
        if win_end > data_len:
            break
    return np.array(DATA)


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
    args.classNum=len(folder_name)
    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的.mat文件名称
        class_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[0]  # 类别的.mat文件路径
        matrix_base = os.path.basename(class_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        class_data = sio.loadmat(class_path)[matrix_name]  # 读入.mat文件

        class_data = np.squeeze(class_data, axis=0)
        class_data_division = sliding_window(class_data) #n*window_len
        #class_data_normalization = trans_norm(class_data_division)  # 归一化处理
        class_data_normalization = class_data_division

        # 数据复制time_of_copy次
        class_data_picture = []
        for j in range(0, len(class_data_normalization)):
            class_data_one = class_data_normalization[j]
            empty = np.zeros((time_of_copy, len(class_data_one)))
            for k in range(0, len(class_data_one)):
                empty[:, k] = class_data_one[k]
            class_data_picture.append(empty)
        class_data_picture = np.array(class_data_picture)  # 列表转换为数组

        # 设置标签
        label = np.zeros((len(class_data_division), len(folder_name)))
        label[:, i] = 1

        # 划分训练数据集和测试数据集
        x_train = class_data_picture[:int(len(class_data_picture)/2), :]
        x_test = class_data_picture[int(len(class_data_picture)/2):, :]
        y_train = label[:int(len(class_data_picture)/2), :]
        y_test = label[int(len(class_data_picture)/2):, :]

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

    return train_X, train_Y, test_X, test_Y, len(folder_name), folder_name


# 绘制混淆矩阵
def show_confusion_matrix(classes, confusion_matrix):
    plt.figure()
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    config = {"font.family": 'Times New Roman'}  # 设置字体类型
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = confusion_matrix.max() / 2.

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if i == j:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5)  # 显示对应的数字
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.work_dir+'/confusion_matrix.jpg', dpi=300)

    # plt.show()


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
    plt.savefig(args.work_dir+'/training_accuracy.jpg', dpi=300)


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
    plt.savefig(args.work_dir+'/verification_accuracy.jpg', dpi=300)


def run_main(x_train, y_train, x_test, y_test, class_num, folder_name, work_dir, model_name):
    len_train = len(x_train)
    len_test = len(x_test)
    train_shuffle = np.arange(len_train)
    test_shuffle = np.arange(len_test)
    np.random.shuffle(train_shuffle)
    np.random.shuffle(test_shuffle)
    x_train = x_train[train_shuffle, :]
    y_train = y_train[train_shuffle, :]
    x_test = x_test[test_shuffle, :]
    y_test = y_test[test_shuffle, :]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_tensor=None,
                                                         input_shape=(x_train.shape[1], x_train.shape[2], 1),
                                                         pooling=None, classes=class_num))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='lr', patience=3, verbose=1,
                                                                   factor=0.5, min_lr=0.00001)
    # 保存最优模型
    checkpoint = tf.keras.callbacks.ModelCheckpoint(work_dir+'/model/'+model_name+'.hdf5', monitor='val_accuracy',
                                                    verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learning_rate_reduction]
    model.summary()
    h = model.fit(x_train, y_train, batch_size=int(args.batch_size), epochs=int(args.max_epochs), shuffle=True,
              validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=2, validation_freq=1)
    h_parameter = h.history
    train_acc(int(args.max_epochs), h_parameter['accuracy'])
    val_acc(h_parameter['val_accuracy'])
    save_model = tf.keras.models.load_model(work_dir+'/model/'+model_name+'.hdf5')
    Y_test = np.argmax(y_test, axis=1)
    y_pred = save_model.predict_classes(x_test)
    labels = folder_name  # 标签显示
    characteristic_matrix = confusion_matrix(Y_test, y_pred)
    show_confusion_matrix(labels, characteristic_matrix)
    print(classification_report(Y_test, y_pred, digits=4))
    args.valAcc=round(max(h_parameter['val_accuracy'])*100,2)

def convert_h5to_pb(h5Path,pbPath):
    model = tf.keras.models.load_model(h5Path,compile=False)
    # model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)   
    #frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=pbPath[:pbPath.rfind(r"/")],
                      name=pbPath[pbPath.rfind(r"/")+1:],
                      as_text=False)
    ipsN,opsN=str(frozen_func.inputs[0]),str(frozen_func.outputs[0])

    inputNodeName=ipsN[ipsN.find("\"")+1:ipsN.find(":")]
    outputNodeName=opsN[opsN.find("\"")+1:opsN.find(":")]

    inputShapeK=ipsN[ipsN.find("=(")+2:ipsN.find("),")] 
    inputShapeF=re.findall(r"\d+\.?\d*",inputShapeK)
    inputShape=reduce(lambda x, y: x + 'x' + y, inputShapeF)

    return inputNodeName,outputNodeName,inputShape

def convert_hdf5_to_trt(work_dir, model_name, workspace='3072', optBatch='20', maxBatch='100'):
    hdfPath = work_dir+"/model/"+model_name+".hdf5"
    pbPath  = work_dir+"/model/temp.pb"
    oxPath  = work_dir+"/model/temp.onnx"
    trtPath = work_dir+"/"+model_name+'.trt'
    
    try:
        inputNodeName,outputNodeName,inputShape=convert_h5to_pb(hdfPath,pbPath)
        #pb converto onnx
        '''python -m tf2onnx.convert  --input temp.pb --inputs Input:0 --outputs Identity:0 --output temp.onnx --opset 11'''
        os.system("python -m tf2onnx.convert  --input "+pbPath+" --inputs "+inputNodeName+":0 --outputs "+outputNodeName+":0 --output "+oxPath+" --opset 11")
        #onnx converto trt
        '''trtexec --explicitBatch --workspace=3072  --minShapes=Input:0:1x128x64x1 --optShapes=Input:0:20x128x64x1 --maxShapes=Input:0:100x128x64x1 --onnx=temp.onnx --saveEngine=temp.trt --fp16'''
        os.system("trtexec --onnx="+oxPath+" --saveEngine="+trtPath+" --workspace="+workspace+" --minShapes=Input:0:1x"+inputShape+\
        " --optShapes=Input:0:"+optBatch+"x"+inputShape+" --maxShapes=Input:0:"+maxBatch+"x"+str(inputShape)+" --fp16")
    except Exception as e:
        print(e)

def generator_model_documents(args):
    from xml.dom.minidom import Document
    doc = Document()  #创建DOM文档对象
    root = doc.createElement('ModelInfo') #创建根元素
    doc.appendChild(root)
    
    model_type = doc.createElement('TRA_DL')
    #model_type.setAttribute('typeID','1')
    root.appendChild(model_type)

    model_item = doc.createElement(args.model_name+'.trt')
    #model_item.setAttribute('nameID','1')
    model_type.appendChild(model_item)

    model_infos = {
        'name':str(args.model_name),
        'type':'TRA_DL',
        'algorithm':'DenseNet121',
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
    shutil.copy(args.work_dir+"/model/"+args.model_name+".hdf5",os.path.join(args.modeldir,args.model_name+'.hdf5'))
    shutil.copy(args.work_dir+"/"+"confusion_matrix.jpg",os.path.join(args.modeldir,'confusion_matrix.jpg'))
    shutil.copy(args.work_dir+"/"+"training_accuracy.jpg",os.path.join(args.modeldir,'training_accuracy.jpg'))
    shutil.copy(args.work_dir+"/"+"verification_accuracy.jpg",os.path.join(args.modeldir,'verification_accuracy.jpg'))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, class_num, folder_name = read_mat(args.data_dir)
    datasetName = args.data_dir.split("/")[-1]
    
    args.work_dir = args.work_dir+'/'+args.time+'-'+datasetName+'-'+args.model_name
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
        os.makedirs(args.work_dir + '/model')
    args.modeldir = args.modeldir+'/'+args.model_name
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)
    run_main(x_train, y_train, x_test, y_test, class_num, folder_name, args.work_dir, args.model_name)
    convert_hdf5_to_trt(args.work_dir, args.model_name)
    generator_model_documents(args)
    print("Train Ended:")

