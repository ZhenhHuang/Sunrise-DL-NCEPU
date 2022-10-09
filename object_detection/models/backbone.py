import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class DarkNet(nn.Module):
    def __init__(self, in_channel=3):
        super(DarkNet, self).__init__()
        self.block1 = nn.Sequential(
            ConvBlock(in_channel, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            ConvBlock(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            ConvBlock(192, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.block5 = nn.Sequential(
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
        )
        self.block6 = nn.Sequential(
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 2, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1)
        )
        self.out_feature = 7 * 7 * 1024

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


def choose_backbone(backbone: str):
    if backbone == 'darknet':
        return DarkNet()
    elif backbone == 'resnet50':
        resnet50 = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(resnet50.children())[:-1])
        model.out_features = resnet50.fc.in_features
        return model
    elif backbone == 'vgg11':
        vgg11 = models.vgg11(pretrained=True)
        model = nn.Sequential(*list(vgg11.children())[:-1])
        model.out_features = 512 * 7 * 7
        return model
    elif backbone == 'vgg16':
        vgg16 = models.vgg16(pretrained=False)
        model = nn.Sequential(*list(vgg16.children())[:-1])
        model.out_features = 512 * 7 * 7
        return model
    elif backbone == 'vgg16_bn':
        vgg16 = models.vgg16_bn(pretrained=False)
        model = nn.Sequential(*list(vgg16.children())[:-1])
        model.out_features = 512 * 7 * 7
        return model