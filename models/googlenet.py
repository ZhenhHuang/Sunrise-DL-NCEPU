import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channel, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, c1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, c2[0], 1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], 3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, c3[0], 1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], 5, padding=2),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, c4, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.concat([x1, x2, x3, x4], dim=1)
        return x


class GoogleNet(nn.Module):
    def __init__(self, in_channel=3, hidden_channel=64, n_classes=102):
        super(GoogleNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, 3*hidden_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            Inception(3*hidden_channel, 64, (96, 128), (16, 32), 32),
            Inception(256, 128,  (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.layers = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )
        self.project = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.layers(x)
        # print(x.shape)
        x = self.project(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    x = torch.rand(size=(1, 3, 256, 256)).to(device)
    model = GoogleNet().to(device)
    out = model(x)
    print(out.shape)





















