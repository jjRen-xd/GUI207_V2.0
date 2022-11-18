import time
import threading
import numpy as np
from torch.utils.data import DataLoader
from config import *
from model import IncrementalModel
from myNetwork import Network
from signalDataset import EvalDataset
from config import model_path, log_path
from main import args
from sklearn.metrics import classification_report


class Evaluation:
    def __init__(self, all_class, new_class, batch_size):
        super().__init__()
        self.new_class = new_class
        self.model = None
        self.batch_size = batch_size
        self.all_class = all_class

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
        old_dataset = EvalDataset(data_type="old")
        new_dataset = EvalDataset(data_type="new")
        observed_dataset = EvalDataset(data_type="observed")

        old_dataloader = DataLoader(old_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
        new_dataloader = DataLoader(new_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)
        observed_dataloader = DataLoader(observed_dataset, shuffle=False, batch_size=self.batch_size, num_workers=1)

        feature_extractor = IncrementalModel()
        self.model = Network(self.all_class, feature_extractor)
        # res = torch.load(model_path + "increment_" + str(self.all_class) + "_256_save.pt")
        res = torch.load(model_path + "increment_" + str(self.all_class) + "_256_save_reduce.pt")
        self.model.load_state_dict(res['model'])
        self.model.to(device)

        old_oa, _ = self.test(old_dataloader)
        new_oa, _ = self.test(new_dataloader)
        all_oa, pred = self.test(observed_dataloader)

        pred = np.concatenate(pred)

        y_pred = np.argmax(pred, axis=1)
        y_true = observed_dataset.targets
        # print(y_pred.shape, y_true.shape)

        print(classification_report(y_true, y_pred))

        return old_oa, new_oa, all_oa

    

if __name__ == '__main__':
    t = Evaluation(args.all_class, args.all_class - args.old_class, args.batch_size)
    old_oa, new_oa, all_oa = t.evaluate()

    timeArray = time.localtime(time.time())
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

    logFile = open(log_path + "log.txt", 'a')
    logFile.write(str(otherStyleTime) + "\n" +
                  str(args) + "\n" +
                  "Old_OA:" + str(old_oa) + "\n" +
                  "New_OA:" + str(new_oa) + "\n" +
                  "All_OA:" + str(all_oa) + "\n\n")
