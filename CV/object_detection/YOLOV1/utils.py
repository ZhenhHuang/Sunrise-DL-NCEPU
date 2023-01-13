import torch


def calc_IoU(bbox1: torch.Tensor, bbox2: torch.Tensor):
    """shape: [N, 4], [M, 4] -> [N, M]"""
    if bbox1.ndim == 1:
        bbox1 = bbox1.unsqueeze(0)
    if bbox2.ndim == 1:
        bbox2 = bbox2.unsqueeze(0)
    left = torch.max(bbox1[:, 0].unsqueeze(1), bbox2[:, 0].unsqueeze(0))
    right = torch.min(bbox1[:, 2].unsqueeze(1), bbox2[:, 2].unsqueeze(0))
    top = torch.max(bbox1[:, 1].unsqueeze(1), bbox2[:, 1].unsqueeze(0))
    bottom = torch.min(bbox1[:, 3].unsqueeze(1), bbox2[:, 3].unsqueeze(0))
    area1 = ((bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])).unsqueeze(1)
    area2 = ((bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])).unsqueeze(0)
    intersection = (right - left).clip(0.) * (bottom - top).clip(0.)
    return intersection / (area1 + area2 - intersection + 1e-6)


def target_encode(target, S, B, num_classes, H, W):
    boxes = target['boxes']
    labels = target['labels']
    target_tensor = torch.zeros(S, S, 5 * B + num_classes)
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5 / torch.tensor([W, H])
    width_height = (boxes[:, 2:] - boxes[:, :2]) / torch.tensor([W, H])
    i, j = (center[:, [1, 0]] * S).long().T
    x_offset, y_offset = center[:, 0] * S - j, center[:, 1] * S - i
    xywhc = torch.concat([x_offset[:, None], y_offset[:, None], width_height, torch.ones((width_height.shape[0], 1))],
                         dim=-1)
    for w in range(B):
        target_tensor[i, j, w * 5:(w + 1) * 5] = xywhc
    target_tensor[i, j, labels + 5 * B] = 1.

    return target_tensor


def target_decode(target_tensor: torch.Tensor, threshold, S, B, H, W):
    """

    :param target_tensor: [S, S, 5 * B + num_classes] [x, y, w, h, confidence]
    :param S: 7
    :param B: 2
    """
    class_score, class_label = torch.max(target_tensor[:, :, 5 * B:], dim=-1)  # [S, S]
    confidence = target_tensor[:, :, [4 + b * 5 for b in range(B)]]  # [S, S, 2]

    mask = (confidence > threshold)
    mask[confidence == confidence.max()] = True
    i, j, b = torch.where(mask == True)  # i: y, j: x, b: box

    confidence = confidence[i, j, b]
    class_label = class_label[i, j]
    class_score = class_score[i, j]

    boxes = target_tensor[i, j][
        torch.arange(b.shape[0])[:, None], [list(range(5 * x, 5 * x + 4)) for x in b]]  # [num of cell, 4]  4:[x,y,w,h]
    cells = torch.stack([j, i]).T
    center = (cells + boxes[:, :2]) / S
    width_height = boxes[:, 2:]
    box_corner = torch.zeros_like(boxes).to(target_tensor.device)  # x1, y1, x2, y2
    box_corner[:, :2] = (center - width_height / 2) * torch.tensor([W, H]).to(center.device)
    box_corner[:, 2:] = (center + width_height / 2) * torch.tensor([W, H]).to(center.device)

    prob = confidence * class_score
    b = torch.where(prob > 0.1)
    if len(b[0]) == 0:
        device = target_tensor.device
        return torch.zeros((1, 4)).to(device), torch.tensor([-1], device=device), torch.zeros(1).to(device), torch.zeros(1).to(device)
    return box_corner[b], class_label[b], confidence[b], class_score[b]


def NonMaximalSuppression(boxes, classes, confidence, scores, threshold):
    """get off overlap boxes"""
    class_unique = torch.unique(classes)
    return_boxes = []
    return_classes = []
    return_probs = []
    for cls in class_unique:
        idx = (classes == cls)
        boxes_masked = boxes[idx]
        prob_masked = scores[idx] * confidence[idx]

        order_idx = torch.argsort(prob_masked, descending=True)
        boxes_masked = boxes_masked[order_idx]
        prob_masked = prob_masked[order_idx]    # ordered descending

        maintain_boxes = []
        maintain_probs = []
        while True:
            if len(boxes_masked) == 0:
                break
            max_box = boxes_masked[0]
            max_prob = prob_masked[0]
            maintain_boxes.append(max_box.reshape(1, -1))
            maintain_probs.append(max_prob.reshape(-1))
            boxes_masked = boxes_masked[1:]
            prob_masked = prob_masked[1:]
            iou = calc_IoU(max_box, boxes_masked)
            maintain_mask = (iou <= threshold).reshape(-1)
            boxes_masked = boxes_masked[maintain_mask]
            prob_masked = prob_masked[maintain_mask]

        maintain_boxes = torch.concat(maintain_boxes, dim=0)
        maintain_probs = torch.concat(maintain_probs, dim=0)

        return_boxes.append(maintain_boxes)
        return_classes.append(torch.tensor([cls] * len(maintain_probs)))
        return_probs.append(maintain_probs)

    return_boxes = torch.vstack(return_boxes)
    return_classes = torch.concat(return_classes, dim=-1)
    return_probs = torch.concat(return_probs, dim=-1)

    return return_boxes, return_classes, return_probs



