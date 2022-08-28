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