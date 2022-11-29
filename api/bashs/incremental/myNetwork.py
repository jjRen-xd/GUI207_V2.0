import torch.nn as nn
import torch
import numpy as np


class Network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(Network, self).__init__()
        self.feature = feature_extractor  # resnet
        self.fc = nn.Linear(feature_extractor.fc.out_features, numclass, bias=False)  # 最后的全连接层

    def forward(self, input):
        x = self.feature(input)
        # print("feature:",x.shape, self.fc.in_features, self.fc.out_features)
        x = self.fc(x)
        return x  # 分类结果

    def get_feature(self, input):
        x = self.feature(input)
        return x

    def Incremental_learning(self, numclass, new_weight=None):
        weight = self.fc.weight.data
        # bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=False)
        self.fc.weight.data[:out_feature] = weight
        if new_weight is not None:
            add_weight = torch.Tensor(new_weight).t()
            self.fc.weight.data[out_feature:] = add_weight
        # self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        return self.feature(inputs)
