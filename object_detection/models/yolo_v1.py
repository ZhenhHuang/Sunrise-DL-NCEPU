import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        # self.norm = nn.BatchNorm2d(c_out)

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
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 2, 1)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 1, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


class YOLO_V1(nn.Module):
    def __init__(self, backbone: str, S, B, num_classes, dropout=0.5):
        super(YOLO_V1, self).__init__()
        self.backbone = self._choose_backbone(backbone)
        self.grids = S
        self.boxes = B
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((S, S))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(S * S * 1024, 4096),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S**2 * (5 * B + num_classes)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x.reshape(-1, self.grids, self.grids, 5 * self.boxes + self.num_classes)

    def _choose_backbone(self, backbone: str):
        if backbone == 'darknet':
            return DarkNet()


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


if __name__ == '__main__':
    model = YOLO_V1('darknet', 7, 2, 20)
    # model = DarkNet()
    x = torch.randn(32, 3, 448, 448)
    out = model(x)
    print(out.shape)