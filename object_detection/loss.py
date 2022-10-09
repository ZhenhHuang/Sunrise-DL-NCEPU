import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calc_IoU


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, num_classes=20, w_coord=5, w_noobj=0.5, **kwargs):
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
        N = preds.shape[0]
        mask = (trues[:, :, :, 4] == 1)  # N, S, S
        loss_noobj = self._loss_noobj(preds, trues, ~mask)  # loss for no obj

        pred_obj = preds[mask]     # pred boxes contain object, [num_obj, 5 * B + num_classes]
        box_pred = pred_obj[:, :self.B * 5].reshape(-1, 5)  # [num_obj_B, 5]
        cls_pred = pred_obj[:, self.B * 5:]

        true_obj = trues[mask]     # true boxes contain object, [num_obj, 5 * B + num_classes]
        box_true = true_obj[:, :self.B * 5].reshape(-1, 5)
        cls_true = true_obj[:, self.B * 5:]

        """compute loss for boxes with object"""
        response_mask = torch.zeros(box_pred.shape[0], dtype=bool).to(pred_obj.device)
        target_iou = torch.zeros(box_true.shape[0]).to(true_obj.device)
        for i in range(0, true_obj.shape[0], self.B):
            pred = box_pred[i: i + self.B]  # [B, 5]
            pred_corner = torch.zeros(pred.shape[0], 4).to(pred_obj.device)     # [B, 4]
            pred_corner[:, :2] = pred[:, :2] - 0.5 * pred[:, 2:4]
            pred_corner[:, 2:] = pred[:, :2] + 0.5 * pred[:, 2:4]

            true = box_true[i].unsqueeze(0)  # [1, 5]
            true_corner = torch.zeros(true.shape[0], 4).to(true_obj.device)  # [1, 4]
            true_corner[:, :2] = true[:, :2] - 0.5 * true[:, 2:4]
            true_corner[:, 2:] = true[:, :2] + 0.5 * true[:, 2:4]

            iou = calc_IoU(pred_corner, true_corner)    # [B, 1]
            max_iou, max_idx = torch.max(iou, dim=0)
            response_mask[i+max_idx] = True
            target_iou[i+max_idx] = max_iou

        box_pred_response = box_pred[response_mask].reshape(-1, 5)
        box_pred_no_response = box_pred[~response_mask].reshape(-1, 5)

        box_true_response = box_true[response_mask].reshape(-1, 5)
        box_true_no_response = box_true[~response_mask].reshape(-1, 5)

        box_true_no_response[:, 4] = 0

        loss_xy = F.mse_loss(box_pred_response[:, :2], box_true_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(box_pred_response[:, 2:4].sqrt(), box_true_response[:, 2:4].sqrt(), reduction='sum')
        loss_obj = F.mse_loss(box_pred_response[:, 4], box_true_response[:, 4], reduction='sum') + \
            F.mse_loss(box_pred_no_response[:, 4], box_true_no_response[:, 4], reduction='sum')
        loss_class = F.mse_loss(cls_pred, cls_true, reduction='sum')

        loss = self.w_coord * (loss_xy + loss_wh) + loss_obj + self.w_noobj * loss_noobj + loss_class
        loss = loss / N
        return loss

    def _loss_noobj(self, preds, trues, mask):
        pred_noobj = preds[mask]
        true_noobj = trues[mask]
        # compute loss of the grids without object
        loss_noobj = F.mse_loss(pred_noobj[:, [4 + b * 5 for b in range(self.B)]],
                                true_noobj[:, [4 + b * 5 for b in range(self.B)]], reduction='sum')
        return loss_noobj


if __name__ == '__main__':
    loss = YOLOLoss()
    x = torch.randn(32, 7, 7, 30)
    y = torch.randn(32, 7, 7, 30)
    output = loss(x, y)
    print(output.shape)


