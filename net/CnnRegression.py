import torch.nn as nn

from net.resnet50 import ResidualBlock, make_layer


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
