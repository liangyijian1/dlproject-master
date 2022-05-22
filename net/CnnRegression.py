import torch.nn as nn
from net.resnet50 import ResidualBlock, make_layer
import torch.nn.functional as F

class ResNet50Regression(nn.Module):

    def __init__(self, inChannel, Block=ResidualBlock) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannel, 64, (7, 7), (2, 2), 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxPool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.layer1 = make_layer(inChannel=64, outChannel=256, block=Block, num_blocks=2, isFirst=True)
        self.layer2 = make_layer(inChannel=256, outChannel=512, block=Block, num_blocks=2)
        self.layer3 = make_layer(inChannel=512, outChannel=1024, block=Block, num_blocks=2)
        self.layer4 = make_layer(inChannel=1024, outChannel=2048, block=Block, num_blocks=2)
        self.avgPool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxPool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        return out


class mlpRegression(nn.Module):
    # def __init__(self, inchannel):
    #     super(mlpRegression, self).__init__()
    #     self.layer1 = nn.Sequential(
    #         nn.Linear(in_features=inchannel, out_features=512),
    #         nn.ReLU()
    #     )
    #     self.layer2 = nn.Sequential(
    #         nn.Linear(in_features=512, out_features=1024),
    #         nn.ReLU()
    #     )
    #     self.layer3 = nn.Sequential(
    #         nn.Linear(in_features=1024, out_features=1)
    #     )
    #
    # def forward(self, x):
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     return out
    def __init__(self):
        super(mlpRegression, self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=256, out_features=1024, bias=True)  # 8*100 8个属性特征
        # 定义第二个隐藏层
        self.hidden2 = nn.Linear(1024, 1024)  # 100*100
        # 定义第三个隐藏层
        self.hidden3 = nn.Linear(1024, 512)  # 100*50
        # 回归预测层
        self.predict = nn.Linear(512, 1)  # 50*1  预测只有一个 房价

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output[:, 0]
