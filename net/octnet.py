import torch.nn as nn
import torch.nn.functional as F


class octnet(nn.Module):

    def __init__(self, num_class, inchannel=1, check_fc=True):
        super().__init__()
        self.check_fc = check_fc
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=128, kernel_size=11, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.SELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.SELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.SELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.SELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.SELU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.SELU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16384, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.flatten(out)
        if self.check_fc:
            out = self.fc1(out)
            out = self.fc2(out)
        # TODO Ìí¼ÓORMÄ£¿é

        return out
