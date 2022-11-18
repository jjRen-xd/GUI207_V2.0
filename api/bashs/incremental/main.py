# coding=utf-8
import time,sys
from config import log_path
from dataProcess import read_txt, split_test_and_train, prepare_pretrain_data, prepare_increment_data, create_dir, read_mat_new
from train import Pretrain, IncrementTrain, Evaluation
import argparse
from utils import generator_model_documents

argparser = argparse.ArgumentParser()

argparser.add_argument('--raw_data_path', type=str, help='the directory where the traing dataset')
argparser.add_argument('--snr', type=int,  help='-2 -4 ... 16 18', default=2)
argparser.add_argument('--pretrain_epoch', type=int,  help='preTrain epoch number,must biger than 1', default=3)
argparser.add_argument('--increment_epoch', type=int,  help='train new model epoch number,must biger than 1', default=5)
argparser.add_argument('--learning_rate', type=float, help='preTrain learning rate', default=1e-4)
argparser.add_argument('--task_size', type=int, help='number of incremental class', default=1)
argparser.add_argument('--old_class', type=int, help='number of old class', default=5)
argparser.add_argument('--all_class', type=int, help='number of all class', default=6)
argparser.add_argument('--memory_size', type=int, help='memory size', default=2000)
argparser.add_argument('--batch_size', type=int, help='batch size', default=32)
argparser.add_argument('--bound', type=float, help='up bound of new class weights', default=0.3)
argparser.add_argument('--random_seed', type=int, help='numpy random seed', default=2022)
argparser.add_argument('--reduce_sample', type=float, help='reduce the number of sample to n%', default=1.0)
argparser.add_argument('--data_dimension', type=int,  help='[39, 128, 256]', default=128)
argparser.add_argument('--test_ratio', type=float, help='the ratio of test dataset', default=0.5)
argparser.add_argument('--work_dir', help='the directory of the training data',default="../../../db/trainLogs")
argparser.add_argument('--time', help='the directory of the training data',default="2022-09-21-21-52-17")
argparser.add_argument('--model_name', help='the directory of the training data',default="model")
argparser.add_argument('--modeldir', help="model saved path", default="../../../db/models")

args = argparser.parse_args()

if __name__ == '__main__':
    import  os
    current_path = os.path.dirname(__file__)
    os.chdir(current_path)
    create_dir()

    # 读取路径下所有文件夹的名称并保存
    folder_path = args.raw_data_path # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_names = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_names.append(file_name[i])
    folder_names.sort()  # 按文件夹名进行排序
    args.classNum=len(folder_names)
    read_mat_new(args.raw_data_path,folder_names)


    datasetName = args.raw_data_path.split("/")[-1]
    args.work_dir = args.work_dir+'/'+args.time+'-'+datasetName+'-'+args.model_name
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
        os.makedirs(args.work_dir + '/model')

    # 开始旧类训练
    pretrain_start = time.time()
    if args.pretrain_epoch != 0:
        preTrain = Pretrain(args.old_class, args.memory_size, args.pretrain_epoch, args.batch_size, args.learning_rate, args.data_dimension)
        preTrain.train()
    pretrain_end = time.time()
    print("pretrain_consume_time:", pretrain_end-pretrain_start)
    sys.stdout.flush()
    # 开始增量训练
    increment_start = time.time()
    if args.increment_epoch != 0:
        incrementTrain = IncrementTrain(args.memory_size, args.all_class, args.all_class-args.old_class, args.task_size, \
        args.increment_epoch, args.batch_size, args.learning_rate, args.bound, args.reduce_sample, args.work_dir, folder_names, args.data_dimension)
        incrementTrain.train()
    increment_end = time.time()
    print("pretrain_consume_time:", increment_end-increment_start)
    sys.stdout.flush()
    valacc=incrementTrain.result.item()
    args.accuracy=round(valacc,2)    #增量训练在测试集上的最高准确率
    # 测试
    evaluation = Evaluation(args.all_class, args.all_class - args.old_class, args.batch_size, args.data_dimension)
    old_oa, new_oa, all_oa, metric = evaluation.evaluate()

    timeArray = time.localtime(time.time())
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

    logFile = open(log_path + "log.txt", 'a')
    logFile.write("\n" + str(otherStyleTime) + "\n" +
                  str(args) + "\n" +
                  "pretrain_consume_time: " + str(pretrain_end-pretrain_start) + "\n" +
                  "increment_consume_time: " + str(increment_end-increment_start) + "\n" +
                  "Old_OA:" + str(old_oa) + "\n" +
                  "New_OA:" + str(new_oa) + "\n" +
                  "All_OA:" + str(all_oa) + "\n\n" + str(metric))
    
    cmd_onnx2trt="trtexec.exe --explicitBatch --workspace=3072 --minShapes=input:1x1x"+\
        str(args.data_dimension)+"x1 --optShapes=input:20x1x"+\
        str(args.data_dimension)+"x1 --maxShapes=input:512x1x"+\
        str(args.data_dimension)+"x1 --onnx="+args.work_dir + \
        "/model/incrementModel.onnx "+" --saveEngine="+\
        args.work_dir + "/"+ args.model_name+".trt --fp16"
    os.system(cmd_onnx2trt)

    args.modeldir = args.modeldir+'/'+args.model_name
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)
    generator_model_documents(args)
    print("Train Ended")
    sys.stdout.flush()

