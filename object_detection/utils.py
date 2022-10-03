import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_IoU(bbox1, bbox2, format='corner'):
    """
    size: [B, 4]
    :param format: 'corner' or 'center'. (x1, y1, x2, y2) or (x, y, w, h)
    :return: IoU value
    """
    def func():
        if format == 'corner':
            x1, y1, x2, y2 = bbox1
            m1, n1, m2, n2 = bbox2
        else:
            x1, y1, x2, y2 = center2corner(bbox1)
            m1, n1, m2, n2 = center2corner(bbox2)

        left = max(m1, x1)
        right = min(m2, x2)
        top = max(n1, y1)
        bottom = min(n2, y2)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (m2 - m1) * (n2 - n1)
        intersection = (right - left) * (bottom - top)
        return intersection / (area1 + area2 - intersection)
    areas = []

    return


def center2corner(bbox):
    x, y, w, h = bbox
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    return x1, y1, x2, y2


def corner2center(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    return x, y, w, h


def target_encode(target, S, B, num_classes, H, W):
    boxes = target['boxes']
    labels = target['labels']
    target_tensor = torch.zeros(S, S, 5 * B + num_classes)
    for k in range(boxes.shape[0]):
        center = (boxes[k, :2] + boxes[k, 2:]) / torch.tensor([W, H]) / 2
        width, height = (boxes[k, 2:] - boxes[k, :2]) / torch.tensor([W, H])
        i, j = int(center[1] * S), int(center[0] * S)
        x_offset, y_offset = center[0] * S - j, center[1] * S - i
        for w in range(B):
            target_tensor[i, j, w*5:(w+1) * 5] = torch.tensor([x_offset, y_offset, width, height, 1.])
        target_tensor[i, j, labels[k] + 5 * B] = 1
    return target_tensor


if __name__ == '__main__':
    print(calc_IoU([2, 2, 4, 4], [3, 3, 5, 5]))