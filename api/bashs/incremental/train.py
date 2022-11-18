# coding=utf-8
import torch,sys
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from model import IncrementalModel, vgg_16, AlexNet

# 三种不同维度的数据，导入对应的模型
from model import AlexNet_256 as IncrementalModel_256
from model import AlexNet_128 as IncrementalModel_128
from model import AlexNet_39 as IncrementalModel_39

from myNetwork import Network
from signalDataset import PretrainSignalDataset, IncrementSignalDataset, EvalDataset
from dataProcess import prepare_pretrain_data, prepare_increment_data, prepare_increment_data_reduce
from config import device, data_path, model_path
import numpy as np
import copy
from utils import mutualInfo_ori, get_weight_by_linearProgram, getOneHot, show_confusion_matrix, show_accplot
from sklearn.metrics import classification_report


class Pretrain:
    def __init__(self, oldClassNumber, memorySize, preTrainEpoch, batch_size, learningRate, dataDimension):
        super().__init__()
        self.oldClassNumber = oldClassNumber
        self.memorySize = memorySize
        self.model = None
        self.preTrainEpoch = preTrainEpoch
        self.batchSize = batch_size
        self.learningRate = learningRate
        self.dataDimension = dataDimension

    def compute_loss(self, signals, target, classNumber):
        output = self.model(signals)
        target = getOneHot(target, classNumber)
        target = target.long()
        output, target = output.to(device), target.to(device)
        # classificationLoss = F.cross_entropy(output, target)
        log_prob = torch.nn.functional.log_softmax(output, dim=1)
        classificationLoss = -torch.sum(log_prob * target) / signals.size(0)
        return classificationLoss

    def test(self, testLoader):
        self.model.eval()
        correct, total = 0, 0
        preds = []
        for setp, (signals, labels) in enumerate(testLoader):
            signals = signals.type(torch.FloatTensor)
            signals, labels = signals.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(signals)
            outputs = outputs.cpu()
            preds.append(outputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100. * correct / total
        self.model.train()
        return accuracy, preds

    def train(self):
        if self.preTrainEpoch == 0:
            return
            # res = torch.load(model_path + "/pretrain_" + str(self.oldClassNumber) + ".pt")
            # self.model.load_state_dict(res['model'])
            # self.model.to(device)
            # test_accuracy, _ = self.test(test_dataloader)
            # print("old model old data test accuracy: {}%".format(test_accuracy))
            # return test_accuracy

        prepare_pretrain_data(self.oldClassNumber)
        if (self.dataDimension == 256):
            featureExtractor = IncrementalModel_256()
        elif (self.dataDimension == 128):
            featureExtractor = IncrementalModel_128()
        elif (self.dataDimension == 39):
            featureExtractor = IncrementalModel_39()
        else:
            print("Data dimention is invaild!")
            sys.stdout.flush()
            
        self.model = Network(self.oldClassNumber, featureExtractor)

        train_dataset = PretrainSignalDataset(data_type="train")
        test_dataset = PretrainSignalDataset(data_type="test")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batchSize, num_workers=1)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.batchSize, num_workers=1)

        self.model.to(device)
        self.model.train()
        best_acc = 0
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.learningRate, weight_decay=0.001)
        for i in range(self.preTrainEpoch):
            loss = []
            if i == 20:
                for p in optimizer.param_groups:
                    p['lr'] = self.learningRate / 2.
            if i == 40:
                for p in optimizer.param_groups:
                    p['lr'] = self.learningRate / 4.
            if i == 60:
                for p in optimizer.param_groups:
                    p['lr'] = self.learningRate / 8.
            if i == 80:
                for p in optimizer.param_groups:
                    p['lr'] = self.learningRate / 16.
            for step, (data, label) in enumerate(train_dataloader):
                data = data.type(torch.FloatTensor)
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                loss_value = self.compute_loss(data, label, self.oldClassNumber)
                loss_value.backward()
                loss.append(loss_value.item())
                optimizer.step()
            loss = np.mean(loss)
            print("epoch:{}, loss_value: {}. The best accuray is {}".format(i + 1, loss, best_acc))
            sys.stdout.flush()
            if (i + 1) % 2 == 0:
                test_accuracy, _ = self.test(testLoader=test_dataloader)
                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    state = {'model': self.model.state_dict()}
                    # best_model = '{}_{}_model.pt'.format(i + 1, '%.3f' % best_acc)
                    torch.save(state, model_path + "pretrain_" + str(self.oldClassNumber) + ".pt")
                print('epoch: {} is finished. accuracy is: {}'.format(i + 1, test_accuracy))
                sys.stdout.flush()
        state = {'model': self.model.state_dict()}
        torch.save(state, model_path + "pretrain_" + str(self.oldClassNumber) + ".pt")
        # res = torch.load(model_path + "/pretrain_" + str(oldClassNumber) + ".pt")
        # model.load_state_dict(res['model'])
        self.save_memory(train_dataset, self.oldClassNumber, self.memorySize)

    def save_memory(self, train_dataset, oldClassNumber, memorySize):
        self.model.eval()
        m = int(memorySize / oldClassNumber)
        exemplarData = []
        exemplarLabel = []
        data, label = train_dataset.data, train_dataset.targets
        for i in range(oldClassNumber):
            ith_data = data[label == i]
            ith_label = label[label == i]
            exemplarData.append(ith_data)
            exemplarLabel.append(ith_label)

        # self.exemplar_set, _ = MinMaxUncertainty(self.model, exemplarData, exemplarLabel, self.memorySize)  # 创建该类的样本集
        # self.exemplar_set, _ = RandomPicking(exemplarData, exemplarLabel, self.memorySize, random_seed=self.random_seed)  # 创建该类的样本集
        exemplarData, exemplarLabel = mutualInfo_ori(self.model, exemplarData, exemplarLabel, memorySize)  # 创建该类的样本集
        exemplarData = np.concatenate(exemplarData)
        exemplarLabel = np.concatenate(exemplarLabel)
        np.save("./data/train/memory/data.npy", exemplarData)
        np.save("./data/train/memory/label.npy", exemplarLabel)


class IncrementTrain:
    def __init__(self, memorySize, allClassNumber, newClassNumber, taskSize, incrementEpoch, batchSize, learningRate, bound, reduce_sample, work_dir, folder_names, dataDimension):
        super().__init__()
        self.memorySize = memorySize
        self.newClassNumber = newClassNumber
        self.taskSize = taskSize
        self.result = 0.0
        self.allClassNumber = allClassNumber
        self.old_model = None
        self.model = None
        self.incrementEpoch = incrementEpoch
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.bound = bound
        self.reduce_sample = reduce_sample
        self.work_dir = work_dir
        self.folder_names=folder_names
        self.dataDimension = dataDimension

    def compute_loss(self, model, signals, target, classNumber):
        output = model(signals)  # ( , 20)
        target = getOneHot(target, classNumber)  # ( , 20)
        target = target.long()
        output, target = output.to(device), target.to(device)
        # classificationLoss = F.cross_entropy(output, target)
        log_prob = torch.nn.functional.log_softmax(output, dim=1)
        classificationLoss = -torch.sum(log_prob * target) / signals.size(0)
        return classificationLoss

    def compute_distill_and_classificationLoss(self, signals, target, task):
        classNumber = task[-1] + 1
        output = self.model(signals)  # ( , 20)
        target = getOneHot(target, classNumber)  # ( , 20)
        target = target.long()
        output, target = output.to(device), target.to(device)
        # classificationLoss = F.cross_entropy(output, target)
        log_prob = torch.nn.functional.log_softmax(output, dim=1)
        classificationLoss = -torch.sum(log_prob * target) / signals.size(0)

        old_output = self.old_model(signals)
        old_output = F.softmax(old_output, dim=1)
        # print("old_output:", old_output.shape)
        old_label = torch.zeros(size=(signals.size(0), classNumber))
        old_label[:, :classNumber - len(task)] = old_output
        old_label = old_label.to(device)
        # log_prob = torch.nn.functional.log_softmax(output, dim=1)
        distill_loss = -torch.sum(log_prob * old_label) / signals.size(0)
        lam = (classNumber - len(task)) / classNumber
        return 0.8 * classificationLoss + 0.2 * distill_loss
        # return 0.61 * classificationLoss + 0.39 * distill_loss
        # return lam * classificationLoss + (1 - lam) * distill_loss

    def test(self, testLoader):
        self.model.eval()
        correct, total = 0, 0
        confusion_matrix = np.zeros((self.allClassNumber, self.allClassNumber))

        for setp, (signals, labels) in enumerate(testLoader):
            signals = signals.type(torch.FloatTensor)
            signals, labels = signals.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(signals)
            outputs = outputs.cpu()
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == labels.cpu()).sum()
            total += len(labels)
            for i in range(0, len(predicts)):
                confusion_matrix[labels[i], predicts[i]] += 1
            # print("@@@@@@@@@@@signals.shape=",signals.shape)
            # print("labels.shape=",labels.shape)
            # print("outputs.shape=",outputs.shape)
            # print("predicts.shape=",predicts.shape)
            # print("labels=",labels)
            # print("predicts=",predicts)
            #exit()
        accuracy = 100. * correct / total
        self.model.train()
        return accuracy, confusion_matrix

    def increment_linearProgram(self, old_class, task, train_dataset):
        classNumber = task[-1] + 1
        self.model.eval()
        oldfeature = []
        for i in range(old_class):
            data, _ = train_dataset.get_image_class(i)
            data = torch.Tensor(data).to(device)
            features = self.model.get_feature(data)
            feature_center = torch.mean(features, dim=0, keepdim=True)
            oldfeature.append(feature_center)
        # oldfeature = torch.stack(oldfeature)
        oldfeature = torch.cat(oldfeature, dim=0)
        newfeature = []
        for i in range(old_class, classNumber):
            data, _ = train_dataset.get_image_class(i)
            data = torch.Tensor(data).to(device)
            features = self.model.get_feature(data)
            feature_center = torch.mean(features, dim=0, keepdim=True)
            newfeature.append(feature_center)
        newfeature = torch.cat(newfeature, dim=0)
        weight = self.model.fc.weight.data
        oldfeature = oldfeature.cpu().detach().numpy()
        newfeature = newfeature.cpu().detach().numpy()
        weight = weight.t().cpu().numpy()
        res = get_weight_by_linearProgram(oldfeature, newfeature, weight, self.bound, len(task), feature_dim=512)
        new_weight = res.x
        new_weight = new_weight.reshape((512, len(task)))
        self.model.Incremental_learning(classNumber, new_weight)

    def get_all_task_newClassNumber(self, old_class):
        classes = []
        if self.newClassNumber % self.taskSize == 0:
            for cls in range(old_class, self.allClassNumber, self.taskSize):
                task_cls = []
                for t in range(self.taskSize):
                    task_cls.append(cls + t)
                classes.append(task_cls)
        else:
            rest = self.newClassNumber % self.taskSize
            for cls in range(old_class, self.allClassNumber - rest, self.taskSize):
                task_cls = []
                for t in range(self.taskSize):
                    task_cls.append(cls + t)
                classes.append(task_cls)
            task_cls = []
            for t in range(rest):
                task_cls.append(self.allClassNumber - rest + t)
            classes.append(task_cls)
        return classes

    def train(self):
        # [[7, 8], [9, 10]]
        old_class = self.allClassNumber - self.newClassNumber
        classes = self.get_all_task_newClassNumber(old_class)

        print("classes:", classes)
        sys.stdout.flush()
        # 更换模型
        if (self.dataDimension == 256):
            featureExtractor = IncrementalModel_256()
        elif (self.dataDimension == 128):
            featureExtractor = IncrementalModel_128()
        elif (self.dataDimension == 39):
            featureExtractor = IncrementalModel_39()
        else:
            print("Data dimention is invaild!")
            sys.stdout.flush()
            
        self.model = Network(old_class, featureExtractor)
        res = torch.load(model_path + "pretrain_" + str(old_class) + ".pt")
        self.model.load_state_dict(res['model'])
        self.model.to(device)

        # 共需增量len(classes)次
        for task in classes:
            num_class = task[-1] + 1
            old_class = num_class - len(task)
            self.old_model = copy.deepcopy(self.model)
            prepare_increment_data_reduce(task,self.reduce_sample)
            # if self.reduce_sample:
            #     # 新类样本量降低80%
            #     prepare_increment_data_reduce(task)
            # else:
            #     # 原始样本量
            #     prepare_increment_data(task)

            train_dataset = IncrementSignalDataset(data_type="train")
            # test_dataset = IncrementSignalDataset(snr=self.snr, data_type="test", dataset=self.dataset)
            test_dataset = EvalDataset(data_type="observed")
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batchSize, num_workers=1)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.batchSize, num_workers=1)

            self.increment_linearProgram(old_class, task, train_dataset)

            self.model.to(device)
            self.model.train()
            best_acc = 0
            acc_list = []
            optimizer = optim.Adam(params=self.model.parameters(), lr=self.learningRate, weight_decay=0.001)
            for i in range(self.incrementEpoch):
                loss = []
                if i == 20:
                    for p in optimizer.param_groups:
                        p['lr'] = self.learningRate / 2.
                if i == 40:
                    for p in optimizer.param_groups:
                        p['lr'] = self.learningRate / 4.
                if i == 60:
                    for p in optimizer.param_groups:
                        p['lr'] = self.learningRate / 8.
                if i == 80:
                    for p in optimizer.param_groups:
                        p['lr'] = self.learningRate / 16.
                for step, (data, label) in enumerate(train_dataloader):
                    #print("label==",label)
                    data = data.type(torch.FloatTensor)
                    data, label = data.to(device), label.to(device)
                    optimizer.zero_grad()
                    loss_value = self.compute_distill_and_classificationLoss(data, label, task)
                    loss_value.backward()
                    loss.append(loss_value.item())
                    optimizer.step()
                loss = np.mean(loss)
                
                print("epoch:{}, loss_value: {}. The best accuray is {}".format(i + 1, loss, best_acc))
                sys.stdout.flush()


                test_accuracy, confusion_matrix = self.test(test_dataloader)
                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    state = {'model': self.model.state_dict()}
                    # best_model = '{}_{}_model.pt'.format(i + 1, '%.3f' % best_acc)
                    # torch.save(state, model_path + "increment_" + str(num_class) + ".pt")
                    torch.save(state, model_path + "increment_" + str(num_class) + ".pt")
                    torch.save(state, self.work_dir + "/model/"+"incrementModel.pt")

                    onnx_save_path = self.work_dir + "/model/"+"incrementModel.onnx"
                    example_tensor = torch.randn(1, 1, self.dataDimension, 1).to(device)
                    torch.onnx.export(self.model,  # model being run
                        example_tensor,  # model input (or a tuple for multiple inputs)
                        onnx_save_path,
                        verbose=False,  # store the trained parameter weights inside the model file
                        training=False,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output']
                        )
                    show_confusion_matrix(self.folder_names,confusion_matrix,self.work_dir)
                if (i + 1) % 2 == 0:
                    print('epoch: {} is finished. accuracy is: {}'.format(i + 1, test_accuracy))
                    sys.stdout.flush()
                acc_list.append(test_accuracy)
            # torch.save(state, model_path + "increment_" + str(num_class) + ".pt")
            show_accplot(len(acc_list),acc_list,self.work_dir)
            # res = torch.load(model_path + "/pretrain_" + str(oldClassNumber) + ".pt")
            # model.load_state_dict(res['model'])
            self.save_memory(train_dataset, num_class, self.memorySize)
            self.result = best_acc

    def save_memory(self, train_dataset, num_class, memorySize):
        self.model.eval()  # 首先将模型转到推理模式
        m = int(memorySize / num_class)  # 内存大小/类别总数 = 每个类别的样本个数
        exemplarData = []
        exemplarLabel = []
        for i in range(num_class):
            data, label = train_dataset.get_image_class(i)
            exemplarData.append(data)
            exemplarLabel.append(label)

        # self.exemplar_set, _ = MinMaxUncertainty(self.model, exemplarData, exemplarLabel, self.memorySize)  # 创建该类的样本集
        # self.exemplar_set, _ = RandomPicking(exemplarData, exemplarLabel, self.memorySize, random_seed=self.random_seed)  # 创建该类的样本集
        exemplarData, exemplarLabel = mutualInfo_ori(self.model, exemplarData, exemplarLabel,
                                                     memorySize)  # 创建该类的样本集
        exemplarData = np.concatenate(exemplarData)
        exemplarLabel = np.concatenate(exemplarLabel)
        np.save(data_path + "train/memory/data.npy", exemplarData)
        np.save(data_path + "train/memory/label.npy", exemplarLabel)


class Evaluation:
    def __init__(self, all_class, new_class, batch_size, data_dimension):
        super().__init__()
        self.new_class = new_class
        self.model = None
        self.batch_size = batch_size
        self.all_class = all_class
        self.data_dimension = data_dimension

    def get_one_hot(self, target, num_class):
        one_hot = torch.zeros(target.shape[0], num_class).to(device)
        one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
        return one_hot

    def compute_loss(self, imgs, target, numclass):
        output = self.model(imgs)  # ( , 20)
        target = self.get_one_hot(target, numclass)  # ( , 20)
        target = target.long()
        output, target = output.to(device), target.to(device)
        # class_loss = F.cross_entropy(output, target)
        log_prob = torch.nn.functional.log_softmax(output, dim=1)
        class_loss = -torch.sum(log_prob * target) / imgs.size(0)
        return class_loss

    def test(self, testloader):
        self.model.eval()  # 切换推理模式
        correct, total = 0, 0
        preds = []
        for setp, (imgs, labels) in enumerate(testloader):
            imgs = imgs.type(torch.FloatTensor)
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs)
            outputs = outputs.cpu()
            preds.append(outputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100. * correct / total
        self.model.train()
        return accuracy, preds

    def evaluate(self):
        print("evaluate...")
        sys.stdout.flush()
        old_dataset = EvalDataset(data_type="old")
        new_dataset = EvalDataset(data_type="new")
        observed_dataset = EvalDataset(data_type="observed")

        old_dataloader = DataLoader(old_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
        new_dataloader = DataLoader(new_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
        observed_dataloader = DataLoader(observed_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)

        if (self.data_dimension == 256):
            feature_extractor = IncrementalModel_256()
        elif (self.data_dimension == 128):
            feature_extractor = IncrementalModel_128()
        elif (self.data_dimension == 39):
            feature_extractor = IncrementalModel_39()
        else:
            print("Data dimention is invaild!")
            sys.stdout.flush()

        self.model = Network(self.all_class, feature_extractor)
        res = torch.load(model_path + "increment_" + str(self.all_class) + ".pt")
        # res = torch.load(model_path + "increment_" + str(self.all_class) + "_256_save.pt")
        # res = torch.load(model_path + "increment_" + str(self.all_class) + "_256_save_reduce.pt")
        self.model.load_state_dict(res['model'])
        self.model.to(device)

        old_oa, _ = self.test(old_dataloader)
        new_oa, _ = self.test(new_dataloader)
        all_oa, pred = self.test(observed_dataloader)

        pred = np.concatenate(pred)

        y_pred = np.argmax(pred, axis=1)
        y_true = observed_dataset.targets
        # print(y_pred.shape, y_true.shape)

        metric = classification_report(y_true, y_pred)

        return old_oa, new_oa, all_oa, metric


if __name__ == '__main__':
    snr = 12
    oldClassNumber = 9
    memorySize = 200
    # preTrain = Pretrain(snr, oldClassNumber, memorySize, dataset="RML2016.04c")
    # preTrain.train()

    newClassNumber = 2
    taskSize = 2
    incrementTrain = IncrementTrain(memorySize, newClassNumber, taskSize)
    incrementTrain.train()
