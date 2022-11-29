import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchsummary
import torchstat
# import ptflops
import numpy as np


# class IncrementalModel(nn.Module):
#     def __init__(self):
#         super(IncrementalModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 1), stride=(1, 1))
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(1, 1))
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1))
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 1), stride=(1, 1))

#         self.bnconv1 = nn.BatchNorm2d(64)
#         self.bnconv2 = nn.BatchNorm2d(128)
#         self.bnconv3 = nn.BatchNorm2d(256)
#         self.bnconv4 = nn.BatchNorm2d(512)

#         self.dropout = nn.Dropout(0.5)

#         self.maxpool = nn.MaxPool2d((3, 1), return_indices=True)

#         self.fc = nn.Linear(512 * 5, 512)

#         # self.cls = nn.Linear(512, numclass)

#     def forward(self, x):
#         x, _ = self.maxpool(F.relu(self.bnconv1(self.conv1(x))))
#         x, _ = self.maxpool(F.relu(self.bnconv2(self.conv2(x))))
#         x, _ = self.maxpool(F.relu(self.bnconv3(self.conv3(x))))
#         x, _ = self.maxpool(F.relu(self.bnconv4(self.conv4(x))))
#         # print(x.shape)
#         x = x.view(-1, 512 * 5)
#         x = self.dropout(F.relu(self.fc(x)))
#         # x = self.cls(x)
#         return x

# (256, 1)
class IncrementalModel(nn.Module):
    def __init__(self):
        super(IncrementalModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 1), stride=(1, 1))

        self.bnconv1 = nn.BatchNorm2d(64)
        self.bnconv2 = nn.BatchNorm2d(128)
        self.bnconv3 = nn.BatchNorm2d(256)
        self.bnconv4 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout(0.5)

        self.maxpool = nn.MaxPool2d((3, 1), return_indices=True)

        self.fc = nn.Linear(512 * 2, 512)

        # self.cls = nn.Linear(512, numclass)

    def forward(self, x):
        x, _ = self.maxpool(F.relu(self.bnconv1(self.conv1(x))))
        x, _ = self.maxpool(F.relu(self.bnconv2(self.conv2(x))))
        x, _ = self.maxpool(F.relu(self.bnconv3(self.conv3(x))))
        x, _ = self.maxpool(F.relu(self.bnconv4(self.conv4(x))))
        # print("output", x.shape)
        x = x.view(-1, 512 * 2)
        x = self.dropout(F.relu(self.fc(x)))
        # x = self.cls(x)
        return x

# (128, 1)
# class IncrementalModel(nn.Module):
#     def __init__(self):
#         super(IncrementalModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 1), stride=(1, 1))
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1))
#         self.conv3 = nn.Conv2d(256, 512, kernel_size=(3, 1), stride=(1, 1))

#         self.bnconv1 = nn.BatchNorm2d(128)
#         self.bnconv2 = nn.BatchNorm2d(256)
#         self.bnconv3 = nn.BatchNorm2d(512)

#         self.dropout = nn.Dropout(0.5)

#         self.maxpool = nn.MaxPool2d((3, 1), return_indices=True)

#         self.fc = nn.Linear(512 * 3, 512)

#         # self.cls = nn.Linear(512, numclass)

#     def forward(self, x):
#         x, _ = self.maxpool(F.relu(self.bnconv1(self.conv1(x))))
#         x, _ = self.maxpool(F.relu(self.bnconv2(self.conv2(x))))
#         x, _ = self.maxpool(F.relu(self.bnconv3(self.conv3(x))))
        
#         x = x.view(-1, 512 * 3)
#         x = self.dropout(F.relu(self.fc(x)))
#         # x = self.cls(x)
#         return x


class vgg_16(nn.Module):
    def __init__(self):
        super(vgg_16, self).__init__()
        # input_1
        # input_1_1 224*224*3 -> 224*224*64
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, padding=1, kernel_size=(3, 1), stride=1)
        self.relu1_1 = nn.ReLU(inplace=False)

        # input_1_2 224*224*64 -> 224*224*64
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=(3, 1), stride=1)
        self.relu1_2 = nn.ReLU(inplace=False)

        # input_1_3 224*224*64 -> 112*112*64
        self.maxpool1_3 = nn.MaxPool2d(padding=0, stride=2, kernel_size=(2, 1))

        # input_2
        # input_2_1 112*112*64 -> 112*112*128
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=(3, 1), stride=1)
        self.relu2_1 = nn.ReLU(inplace=False)

        # input_2_2 112*112*128 -> 112*112*128
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=(3, 1), stride=1)
        self.relu2_2 = nn.ReLU(inplace=False)

        # input_2_3 112*112*128 -> 56*56*128
        self.maxpool2_3 = nn.MaxPool2d(padding=0, stride=2, kernel_size=(2, 1))

        # input_3
        # input_3_1 56*56*128 -> 56*56*256
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=(3, 1), stride=1)
        self.relu3_1 = nn.ReLU(inplace=False)

        # input_3_2 56*56*256 -> 56*56*256
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=(3, 1), stride=1)
        self.relu3_2 = nn.ReLU(inplace=False)

        # input_3_3 56*56*256 -> 56*56*256
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=(3, 1), stride=1)
        self.relu3_3 = nn.ReLU(inplace=False)

        # input_3_4 56*56*256 -> 28*28*256
        self.maxpool3_4 = nn.MaxPool2d(padding=0, stride=2, kernel_size=(2, 1))

        # input_4
        # input_4_1 28*28*256 -> 28*28*512
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=(3, 1), stride=1)
        self.relu4_1 = nn.ReLU(inplace=False)

        # input_4_2 28*28*512 -> 28*28*512
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=(3, 1), stride=1)
        self.relu4_2 = nn.ReLU(inplace=False)

        # input_4_3 28*28*512 -> 28*28*512
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=(3, 1), stride=1)
        self.relu4_3 = nn.ReLU(inplace=False)

        # input_4_4 28*28*512 -> 14*14*512
        self.maxpool4_4 = nn.MaxPool2d(padding=0, stride=2, kernel_size=(2, 1))

        # input_5
        # input_5_1 14*14*512 -> 14*14*512
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=(3, 1), stride=1)
        self.relu5_1 = nn.ReLU(inplace=False)

        # input_5_2 14*14*512 -> 14*14*512
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=(3, 1), stride=1)
        self.relu5_2 = nn.ReLU(inplace=False)

        # input_5_3 14*14*512 -> 14*14*512
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=(3, 1), stride=1)
        self.relu5_3 = nn.ReLU(inplace=False)

        # input_5_4 14*14*512 -> 7*7*512
        self.maxpool5_4 = nn.MaxPool2d(padding=0, kernel_size=2, stride=(2, 1))

        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(88 * 512, 4096)
        self.relu6 = nn.ReLU(inplace=False)
        self.drouout6 = nn.Dropout(0.5)

        self.linear7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=False)
        self.drouout7 = nn.Dropout(0.5)

        self.fc = nn.Linear(4096, 512)
        self.relu8 = nn.ReLU(inplace=False)

    def forward(self, x):
        # stage 1
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_2(x)
        x = self.maxpool1_3(x)

        # stage 2
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.maxpool2_3(x)

        # stage 3
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.maxpool3_4(x)

        # stage 4
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.maxpool4_4(x)

        # stage 5
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.maxpool5_4(x)

        # stage 6
        x = self.flatten(x)
        # print(x.shape)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.drouout6(x)

        # stage 7
        x = self.linear7(x)
        x = self.relu7(x)
        x = self.drouout7(x)

        # stage 8
        x = self.fc(x)
        x = self.relu8(x)
        return x


class AlexNet_256(nn.Module):
    configs = [1, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
        int(x * width_mult), AlexNet_256.configs))
        super(AlexNet_256, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=(11, 1), stride=(2, 1)),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=(5, 1)),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]
        self.fc = nn.Linear(256 * 22, 512)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 22)
        x = self.fc(x)
        return x

class AlexNet_128(nn.Module):
    configs = [1, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
        int(x * width_mult), AlexNet_128.configs))
        super(AlexNet_128, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=(11, 1), stride=(2, 1)),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=(5, 1)),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]
        self.fc = nn.Linear(256 * 6, 512)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 6)
        x = self.fc(x)
        return x

class AlexNet_39(nn.Module):
    configs = [1, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
        int(x * width_mult), AlexNet_39.configs))
        super(AlexNet_39, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=(11, 1), stride=(1, 1)),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=(5, 1)),
            nn.BatchNorm2d(configs[2]),
            # nn.MaxPool2d(kernel_size=(3, 1), stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=(3, 1)),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]
        self.fc = nn.Linear(256 * 4, 512)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 4)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # model = AcarsModel(20)
    # x = torch.Tensor(np.Random.randn(2, 1, 4096, 2))
    # out = model(x)

    # model = IncrementalModel()
    # x = torch.Tensor(np.random.randn(2, 1, 256, 1))
    # out = model(x)
    # print(out.shape)

    # vgg16 = vgg_16()
    # x = torch.Tensor(np.random.randn(2, 1, 256, 1))
    # out = vgg16(x)
    # print(out.shape)


    
    # alexNet = AlexNet_256()
    # x = torch.Tensor(np.random.randn(2, 1, 256, 1))
    # out = alexNet(x)
    # print(out.shape)

    # alexNet = AlexNet_128()
    # x = torch.Tensor(np.random.randn(2, 1, 128, 1))
    # out = alexNet(x)
    # print(out.shape)

    alexNet = AlexNet_39()
    x = torch.Tensor(np.random.randn(2, 1, 39, 1))
    out = alexNet(x)
    print(out.shape)


    # torchsummary.summary(model.cuda(), (1, 128, 2))

    # torchstat.stat(model, [1, 128, 2])

    # flops, params = ptflops.get_model_complexity_info(model, (1, 4096, 2), as_strings=True)

    # print(flops)
    # print(params)
