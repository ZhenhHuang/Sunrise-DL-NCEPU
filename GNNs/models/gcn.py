import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj, is_sparse=True):
        x = self.linear(x)
        if is_sparse:
            x = torch.spmm(adj, x)
        else:
            x = torch.matmul(adj, x)
        return x


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, dropout=0.5, activation='relu', **kwargs):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GraphConvolution(in_features, hidden_features, bias=True)
        self.conv2 = GraphConvolution(hidden_features, num_classes, bias=True)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, adj):
        x = self.dropout(self.activation(self.conv1(x, adj)))
        x = self.conv2(x, adj)
        return x