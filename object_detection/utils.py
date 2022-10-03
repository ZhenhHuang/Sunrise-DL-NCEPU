import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_IoU(bbox1: torch.Tensor, bbox2: torch.Tensor):
    """shape: [N, 4], [M, 4] -> [N, M]"""
    left = torch.max(bbox1[:, 0].unsqueeze(1), bbox2[:, 0].unsqueeze(0))
    right = torch.min(bbox1[:, 2].unsqueeze(1), bbox2[:, 2].unsqueeze(0))
    top = torch.max(bbox1[:, 1].unsqueeze(1), bbox2[:, 1].unsqueeze(0))
    bottom = torch.min(bbox1[:, 3].unsqueeze(1), bbox2[:, 3].unsqueeze(0))
    area1 = ((bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])).unsqueeze(1)
    area2 = ((bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])).unsqueeze(0)
    intersection = (right - left) * (bottom - top)
    return intersection / (area1 + area2 - intersection)


def target_encode(target, S, B, num_classes, H, W):
    boxes = target['boxes']
    labels = target['labels']
    target_tensor = torch.zeros(S, S, 5 * B + num_classes)
    for k in range(boxes.shape[0]):
        center = (boxes[k, :2] + boxes[k, 2:]) / torch.tensor([W, H]) / 2
        width, height = (boxes[k, 2:] - boxes[k, :2]) / torch.tensor([W, H])
        i, j = int(center[1] * S), int(center[0] * S)
        x_offset, y_offset = center[0] * S - j, center[1] * S - i
        # for w in range(B):
        #     target_tensor[i, j, w*5:(w+1) * 5] = torch.tensor([x_offset, y_offset, width, height, 1.])
        target_tensor[i, j, 0: 5] = torch.tensor([x_offset, y_offset, width, height, 1.])
        target_tensor[i, j, labels[k] + 5 * B] = 1
    return target_tensor


def target_decode(target_tensor: torch.Tensor, threshold, S, B):
    """

    :param target_tensor: [S, S, 5 * B + num_classes] [x, y, w, h, confidence]
    :param S: 7
    :param B: 2
    """
    class_score, class_label = torch.max(target_tensor[:, :, 5 * B:], dim=-1)   # [S, S]
    confidence = target_tensor[:, :, [4 + b * 5 for b in range(B)]] * class_score.unsqueeze(-1)     # [S, S, 2]
    i, j, b = torch.where(confidence >= threshold)   # i: y, j: x, b: box
    confidence = confidence[i, j, b]
    class_label = class_label[i, j]
    class_score = class_score[i, j]

    boxes = target_tensor[i, j][torch.arange(b.shape[0])[:, None], [list(range(5*x, 5*x+4)) for x in b]]    # [num of cell, 4]  4:[x,y,w,h]
    cells = torch.stack([j, i]).T
    center = (cells + boxes[:, :2]) / S
    width_height = boxes[:, 2:]
    box_corner = torch.zeros_like(boxes).to(target_tensor.device)   # x1, y1, x2, y2
    box_corner[:, :2] = center - width_height / 2
    box_corner[:, 2:] = center + width_height / 2

    return box_corner, class_label, confidence, class_score

