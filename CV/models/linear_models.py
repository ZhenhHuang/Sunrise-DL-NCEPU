import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, num_hidden, classes):
        """
        Simple linear classification model. Use softmax to prediction distribution.
        f(x) = WX+b, y = softmax(f(x))
        :param num_hidden: dimension of input data
        :param classes: number of classes for classification
        """
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(num_hidden, classes, bias=True)

    def forward(self, x):
        # [N, H, W, C]
        N = x.shape[0]
        x = self.linear(x.reshape(N, -1))
        x = torch.log_softmax(x, dim=-1)    # N, classes
        return x