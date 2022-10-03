import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calc_IoU


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, num_classes=20, w_coord=5, w_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.w_coord = w_coord
        self.w_noobj = w_noobj

    def forward(self, preds, trues):
        """
        :param preds: [N, S, S, 5 * B + num_classes]
        :param trues: [N, S, S, 5 * B + num_classes]
        """
        mask = (trues[:, :, :, 4] == 1)  # N, S, S
        pred_obj = preds[mask]     # pred boxes contain object, num_obj, 5 * B + num_classes
        box_pred = pred_obj[:, :self.B * 5].reshape(-1, 5)
        cls_pred = pred_obj[:, self.B * 5:]

        true_obj = trues[mask]     # true boxes contain object, num_obj, 5 * B + num_classes
        box_true = true_obj[:, :self.B * 5].reshape(-1, 5)
        cls_true = true_obj[:, self.B * 5:]

        pred_noobj = preds[~mask]
        true_noobj = trues[~mask]

        # compute loss of the grids without object
        loss_noobj = F.mse_loss(pred_noobj[:, [b * 4 for b in range(self.B)]])




