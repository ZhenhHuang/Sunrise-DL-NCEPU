from xml.etree.ElementInclude import include
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, in_channel, out_channel=64, out_dims=10, activation='relu', dropout=0.5, pool='max'):
        """
        A simple convolution network for classification.
        """
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=0)     # 32 -> 30
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=out_channel, kernel_size=3, stride=1, padding=0)    # 30 -> 28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) if pool == 'max' else nn.AvgPool2d(kernel_size=2, stride=2)  # 28 -> 14

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)     # 12 -> 10
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=3, stride=1, padding=0)    # 10 -> 8
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) if pool == 'max' else nn.AvgPool2d(kernel_size=2, stride=2)  # 8 -> 4

        self.project = nn.Linear(14 * 14 * out_channel, out_dims)
        self.activation = F.relu if activation == 'relu' else F.tanh
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool1(x)

        # x = self.activation(self.conv3(x))
        # x = self.activation(self.conv4(x))
        # x = self.pool2(x)

        x = self.drop(self.project(x.reshape(x.shape[0], -1)))
        x = torch.log_softmax(x, dim=-1)
        return x
    

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        if stride > 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.downsample(x) if hasattr(self, 'downsample') else x
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels=3, out_dims=10):
        """_summary_
        ResNet18 for CIFAR-10
        Args:
            in_channels (int, optional): _description_. Defaults to 3.
            out_dims (int, optional): _description_. Defaults to 10.
        """
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.block1 = nn.Sequential(ResNetBlock(16, 32, 2), ResNetBlock(32, 32, 1))
        self.block2 = nn.Sequential(ResNetBlock(32, 64, 2), ResNetBlock(64, 64, 1))
        self.proj = nn.Linear(8 * 8 * 64, out_dims)
    
    def forward(self, x):
        x = self.norm1(self.conv1(x))   # 32 32 64
        x = self.block1(x)
        x = self.block2(x)
        x = self.proj(x.reshape(x.shape[0], -1))
        return torch.log_softmax(x, dim=-1)
        

        