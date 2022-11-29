# -*- coding: UTF-8 -*-
import os,shutil
import sys
import argparse
import tensorflow.compat.v1 as tfv1
import tensorflow.keras as keras
import numpy as np
import abfc_model
from data_process import read_mat, storage_characteristic_matrix, data_normalization
from data_process import show_feature_selection, show_confusion_matrix
from utils import BatchCreate

theone="39"
tfv1.compat.v1.disable_eager_execution()

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])



parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data',default="../db/datasets/falseHRRPmat_1x128")
parser.add_argument('--work_dir', help='the directory of the training data',default="../db/trainLogs")
parser.add_argument('--time', help='the directory of the training data',default="2022-09-21-21-52-17")
parser.add_argument('--model_name', help='the directory of the training data',default="model")
parser.add_argument('--batch_size', help='the number of batch size',default=32)
parser.add_argument('--max_epochs', help='the number of epochs',default=1)
parser.add_argument('--modeldir', help="model saved path", default="../db/models")
args = parser.parse_args()


def test(train_X, train_Y, test_X, test_Y, epoch, output_size, fea_num, work_dir, model_name):
    train_model = keras.models.Sequential([
        keras.layers.Conv1D(64, kernel_size=1, padding='valid', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(output_size, activation='softmax')])
    train_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    # 建立存储保存后的模型的文件夹
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.makedirs(work_dir + '/model')
    save_model_path = work_dir + '/model/'+model_name+'_feature_' + str(fea_num) + '.hdf5'
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.99, patience=3,verbose=0, min_lr=0.0001)
    checkpoint = keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learn_rate_reduction]
    train_model.fit(train_X, train_Y, batch_size=int(args.batch_size), epochs=epoch, shuffle=True,
                   validation_data=(test_X, test_Y), callbacks=callbacks_list, verbose=0, validation_freq=1)
    test_model = keras.models.load_model(save_model_path)
    Y_test = np.argmax(test_Y, axis=1)
    Y_pred = test_model.predict_classes(test_X)

    return Y_test, Y_pred


def run_test(A, train_X, train_Y, test_X, test_Y, epoch, output_size, f_start, f_end, f_interval, work_dir, model_name):
    all_characteristic_matrix = []  # 存储特征矩阵
    attention_weight = A.mean(0)  # [512,]
    ABFC_wight_rank = list(np.argsort(attention_weight))[::-1]
    ac_score_list = []
    for K in range(f_start, f_end + 1, f_interval):
        use_train_x = np.expand_dims(train_X[:, ABFC_wight_rank[:K]], axis=-1)
        use_test_x = np.expand_dims(test_X[:, ABFC_wight_rank[:K]], axis=-1)
        label_class, predicted_class = test(use_train_x, train_Y, use_test_x, test_Y, epoch, output_size, K, work_dir, model_name)
        characteristic_matrix, accuracy_every_class, accuracy = storage_characteristic_matrix(predicted_class, label_class, output_size)

        print('Using Top {} features| accuracy:{:.4f}'.format(K, accuracy))
        sys.stdout.flush()
        all_characteristic_matrix.append(characteristic_matrix)
        ac_score_list.append(accuracy)

    return ac_score_list, all_characteristic_matrix


def run_train(sess, train_X, train_Y, train_step, batch_size):
    X = tfv1.get_collection('input')[0]
    Y = tfv1.get_collection('output')[0]

    Iterator = BatchCreate(train_X, train_Y)
    for step in range(1, train_step + 1):
        if step % 100 == 0:
            val_loss, val_accuracy = sess.run(tfv1.get_collection('validate_ops'), feed_dict={X: train_X, Y: train_Y})
            # val_loss, val_accuracy = sess.run(tfv1.get_collection('validate_ops'), feed_dict={X: val_X, Y: val_Y})

            print('[%4d] ABFC-loss:%.12f ABFC-accuracy:%.6f' % (step, val_loss, val_accuracy))
            sys.stdout.flush()
        xs, ys = Iterator.next_batch(batch_size)
        _, A = sess.run(tfv1.get_collection('train_ops'), feed_dict={X: xs, Y: ys})
    return A


def inference(data_path,train_step, batchsize, f_start, f_end, f_interval, work_dir, model_name):
    train_X, train_Y, test_X, test_Y, class_label = read_mat(data_path)
    args.classNum=len(class_label)
    train_X, test_X = data_normalization(train_X), data_normalization(test_X)
    Train_Size = len(train_X)
    total_batch = Train_Size / batchsize
    abfc_model.build(total_batch, len(train_X[0]), len(train_Y[0]))
    with tfv1.Session() as sess:  # 创建上下文
        tfv1.global_variables_initializer().run()  # 初始化模型参数
        print('== Get feature weight by using ABFC ==')
        sys.stdout.flush()
        A = run_train(sess, train_X, train_Y, train_step, batchsize)
    print('==  The Evaluation of ABFC ==')
    sys.stdout.flush()
    at = A.mean(0)
    A_wight_rank = list(np.argsort(at))[::-1]
    save_A_path = work_dir + '/model/'+'attention.txt'
    attention_weights = open(save_A_path, 'w', encoding='utf-8')
    for i in range(0, len(A_wight_rank)):
        attention_weights.write(str(A_wight_rank[i])+'\n')
    attention_weights.close()
    ac_score_list, characteristic_matrix_summary = run_test(A, train_X, train_Y, test_X,
                                            test_Y, train_step, len(train_Y[0]), f_start, f_end, f_interval, work_dir, model_name)
    show_feature_selection(ac_score_list, f_start, f_end, f_interval, work_dir)#hh
    optimal_result = ac_score_list.index(max(ac_score_list))
    show_confusion_matrix(class_label, characteristic_matrix_summary[optimal_result], work_dir)#hh
    print(optimal_result)
    sys.stdout.flush()
    attention_weights = open(save_A_path, 'a', encoding='utf-8')
    attention_weights.write(str(ac_score_list.index(max(ac_score_list))+1)+'\n')
    attention_weights.close()
    print(max(ac_score_list))
    sys.stdout.flush()
    global theone
    theone=str(ac_score_list.index(max(ac_score_list))+1)

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
        'algorithm':'ABFC',
        'framework':'keras',
        'accuracy':'-',
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

    shutil.copy(args.work_dir+"/"+args.model_name+"_feature_"+str(theone)+".trt",os.path.join(args.modeldir,args.model_name+'.trt'))
    shutil.copy(args.work_dir+"/model/attention.txt",os.path.join(args.modeldir,'/model/attention.txt'))
    shutil.copy(args.work_dir+"/model/"+args.model_name+"_feature_"+str(theone)+".hdf5",os.path.join(args.modeldir,args.model_name+'.hdf5'))
    shutil.copy(args.work_dir+"/"+"confusion_matrix.jpg",os.path.join(args.modeldir,'confusion_matrix.jpg'))
    shutil.copy(args.work_dir+"/"+"features_Accuracy.jpg",os.path.join(args.modeldir,'features_Accuracy.jpg'))

if __name__ == '__main__':
    # try:
    path = args.data_dir
    datasetName = args.data_dir.split("/")[-1]
    args.work_dir = args.work_dir+"/"+args.time+"-"+datasetName+"-"+args.model_name
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
        os.makedirs(args.work_dir + '/model')

    args.modelid=1
    args.schedule=0
    inference(path,int(args.max_epochs), int(args.batch_size), 1, 39, 1, args.work_dir, args.model_name)
    print("theone==",theone)
    sys.stdout.flush()
    #debug#os.system("python ../hdf52trt.py --model_type AFS --work_dir "+args.work_dir+" --model_name "+args.model_name+" --afsmode_Idx " + theone)
    #debug#cmd="python ../hdf52trt.py --model_type AFS --work_dir D:/lyh/GUI207_V2.0/db/trainLogs/2022-10-09-10-17-15-AFS-fea39/ --model_name fea --afsmode_Idx " + theone;
    
    cmd="python ../api/bashs/hdf52trt.py --model_type ABFC --work_dir "+args.work_dir+" --model_name "+args.model_name+" --abfcmode_Idx " + theone
    os.system(cmd)
    #python ../../api/bashs/hdf52trt.py --model_type ABFC --work_dir D:/lyh/GUI207_V2.0/db/trainLogs/2022-09-28-13-51-11-ABFC-falseHRRPmat --model_name trttest --abfcmode_Idx 39
    #convert_hdf5_to_trt('ABFC', args.work_dir, args.model_name,theone)
    #os.system("python D:/lyh/GUI207_V2.0/api/bashs/abfc/test.py")
    args.modeldir = args.modeldir+'/'+args.model_name
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)
    generator_model_documents(args)
    print("Train Ended:")
    sys.stdout.flush()  
    # except Exception as re:
    #     print("Train Failed:",re)
