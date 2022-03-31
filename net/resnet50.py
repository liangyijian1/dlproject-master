import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride=2):
        super(ResidualBlock, self).__init__()
        insideChannel = int(outChannel / 4)
        self.left = nn.Sequential(
            nn.Conv2d(inChannel, insideChannel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(insideChannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(insideChannel, insideChannel, kernel_size=(3, 3), stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(insideChannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(insideChannel, outChannel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(outChannel),
        )
        self.shortcut = nn.Sequential()
        if inChannel != outChannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=(1, 1), stride=stride, padding=0, bias=True),
                nn.BatchNorm2d(outChannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def make_layer(inChannel, outChannel, block, num_blocks, isFirst=False):
    layers = []
    for i in range(num_blocks):
        if isFirst or i == num_blocks - 1:
            layers.append(block(inChannel, outChannel, 1))
        else:
            layers.append(block(inChannel, outChannel))
        if isFirst:
            inChannel *= 4
        else:
            inChannel *= 2

    return nn.Sequential(*layers)


class ResNet50(nn.Module):

    def __init__(self, Block, inChannel, num_class) -> None:
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
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxPool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
