import torch
import torch.nn as nn



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.spmm(adj, x)
        return x


class SparseDropout(nn.Module):
    def __init__(self, prob=0.5):
        super(SparseDropout, self).__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor):
        mask = (torch.rand(x._values().shape) + (1 - self.prob)).floor().bool()
        idx = x._indices()[:, mask]
        value = x._values()[mask] / (1 - self.prob)
        return torch.sparse_coo_tensor(idx, value, x.shape)

