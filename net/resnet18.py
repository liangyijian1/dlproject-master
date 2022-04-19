import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, r):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Linear(in_features=channel, out_features=int(channel / r)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=int(channel / r), out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.gap(x)
        out = out.view(out.size(0), -1)
        out = self.ca(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, stride),
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2, in_channel=1):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.ca1 = ChannelAttention(64, 16)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.ca2 = ChannelAttention(128, 16)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.ca3 = ChannelAttention(256, 16)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.ca4 = ChannelAttention(512, 16)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxPool(out)
        out = self.layer1(out)
        out = self.ca1(out).view(out.size(0), 64, 1, 1) * out
        out = self.layer2(out)
        out = self.ca2(out).view(out.size(0), 128, 1, 1) * out
        out = self.layer3(out)
        out = self.ca3(out).view(out.size(0), 256, 1, 1) * out
        out = self.layer4(out)
        out = self.ca4(out).view(out.size(0), 512, 1, 1) * out
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet18DeepFeature(nn.Module):
    def __init__(self, residualBlock=ResidualBlock, num_classes=2, in_channel=1):
        super(ResNet18DeepFeature, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(residualBlock, 64, 2, stride=1)
        self.ca1 = ChannelAttention(64, 16)
        self.layer2 = self.make_layer(residualBlock, 128, 2, stride=2)
        self.ca2 = ChannelAttention(128, 16)
        self.layer3 = self.make_layer(residualBlock, 256, 2, stride=2)
        self.ca3 = ChannelAttention(256, 16)
        self.layer4 = self.make_layer(residualBlock, 512, 2, stride=2)
        self.ca4 = ChannelAttention(512, 16)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxPool(out)
        out = self.layer1(out)
        out = self.ca1(out).view(out.size(0), 64, 1, 1) * out
        out = self.layer2(out)
        out = self.ca2(out).view(out.size(0), 128, 1, 1) * out
        out = self.layer3(out)
        out = self.ca3(out).view(out.size(0), 256, 1, 1) * out
        out = self.layer4(out)
        out = self.ca4(out).view(out.size(0), 512, 1, 1) * out
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(num_classes):
    return ResNet(num_classes=num_classes)


def init_weight(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
