import torch
import torch.nn as nn
import torch.nn.functional as F
from object_detection.models.backbone import choose_backbone


class YOLO_V1(nn.Module):
    def __init__(self, backbone: str, S, B, num_classes, dropout=0.5):
        super(YOLO_V1, self).__init__()
        self.backbone = choose_backbone(backbone)
        self.grids = S
        self.boxes = B
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_features, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(4096, S**2 * (5 * B + num_classes)),
            nn.Sigmoid()
        )
        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.backbone.out_features, S ** 2 * (5 * B + num_classes)),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x.reshape(-1, self.grids, self.grids, 5 * self.boxes + self.num_classes)


if __name__ == '__main__':
    model = YOLO_V1('resnet50', 7, 2, 20)
    x = torch.randn(32, 3, 448, 448)
    out = model(x)
    print(out.shape)