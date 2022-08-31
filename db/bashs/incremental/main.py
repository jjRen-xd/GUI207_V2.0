# coding=utf-8
import time
from config import log_path
from dataProcess import read_txt, split_test_and_train, prepare_pretrain_data, prepare_increment_data, create_dir, read_mat_39, read_mat_256, read_mat_128_new
from train import Pretrain, IncrementTrain, Evaluation
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--raw_data_path', type=str, help='the directory where the traing dataset')
argparser.add_argument('--snr', type=int,  help='-2 -4 ... 16 18', default=2)
argparser.add_argument('--pretrain_epoch', type=int,  help='preTrain epoch number', default=0)
argparser.add_argument('--increment_epoch', type=int,  help='train new model epoch number', default=50)
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

args = argparser.parse_args()

if __name__ == '__main__':
    import  os

    current_path = os.path.dirname(__file__)

    os.chdir(current_path)


    # create_dir()

    # if args.data_dimension == 39:
    #     read_mat_39(args.raw_data_path)
    # if args.data_dimension == 128:
    #     read_mat_128_new(args.raw_data_path)
    # if args.data_dimension == 256:
    #     read_mat_256_new(args.raw_data_path)

    # txt文件转为npy文件
    # read_txt(args.raw_data_path, args.class_name, args.snr)
    # 分割训练集和测试集
    # split_test_and_train(args.test_ratio, args.random_seed)

    # 开始旧类训练
    pretrain_start = time.time()
    if args.pretrain_epoch != 0:
        preTrain = Pretrain(args.old_class, args.memory_size, args.pretrain_epoch, args.batch_size, args.learning_rate)
        preTrain.train()
    pretrain_end = time.time()
    print("pretrain_consume_time:", pretrain_end-pretrain_start)

    # 开始增量训练
    increment_start = time.time()
    if args.increment_epoch != 0:
        incrementTrain = IncrementTrain(args.memory_size, args.all_class, args.all_class-args.old_class, args.task_size, args.increment_epoch, args.batch_size, args.learning_rate, args.bound, args.reduce_sample)
        incrementTrain.train()
    increment_end = time.time()
    print("pretrain_consume_time:", increment_end-increment_start)

    # 测试
    evaluation = Evaluation(args.all_class, args.all_class - args.old_class, args.batch_size)
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


