import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj, sparse=False):
        x = self.linear(x)
        if sparse:
            x = torch.spmm(adj, x)
        else:
            x = torch.matmul(adj, x)
        return x


class Attentive(nn.Module):
    def __init__(self, in_features):
        super(Attentive, self).__init__()
        self.weights = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return x * self.weights


class SparseDropout(nn.Module):
    def __init__(self, prob=0.5):
        super(SparseDropout, self).__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor):
        mask = (torch.rand(x._values().shape) + (1 - self.prob)).floor().bool()
        idx = x._indices()[:, mask]
        value = x._values()[mask] / (1 - self.prob)
        return torch.sparse_coo_tensor(idx, value, x.shape)