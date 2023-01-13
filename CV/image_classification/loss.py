import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialLoss(nn.Module):
    def __init__(self, device):
        super(SpecialLoss, self).__init__()
        self.w1 = nn.Parameter(torch.tensor(1.)).to(device)
        self.w2 = nn.Parameter(torch.tensor(1.)).to(device)
        self.w3 = nn.Parameter(torch.tensor(1.)).to(device)

    def forward(self, x1, x2, x3, y1, y2, y3):
        loss1 = F.cosine_similarity(x1, x2).sum()
        loss2 = F.cosine_similarity(x1, x3).sum()
        loss3 = F.cross_entropy(x1, y1)
        loss = self.w1 * loss1 - self.w2 * loss2 + self.w3 * loss3
        return loss


class DistilLoss(nn.Module):
    def __init__(self, T, alpha):
        super(DistilLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.kldloss = nn.KLDivLoss()

    def forward(self, y1, y2, y):
        loss1 = F.cross_entropy(y1, y)
        st = torch.log_softmax(y1 / self.T, dim=-1)
        tch = torch.softmax(y2 / self.T, dim=-1)
        loss2 = self.kldloss(st, tch) * self.T ** 2
        loss = (1 - self.alpha) * loss1 + self.alpha * loss2
        return loss