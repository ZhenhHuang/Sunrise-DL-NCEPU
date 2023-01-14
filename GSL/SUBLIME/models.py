import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, SparseDropout


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers, dropout_node=0.5, dropout_edge=0.25):
        super(GCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConvolution(in_features, hidden_features))
        for _ in range(n_layers-2):
            self.conv_layers.append(GraphConvolution(hidden_features, hidden_features))
        self.conv_layers.append(GraphConvolution(hidden_features, out_features))
        self.dropout_node = nn.Dropout(dropout_node)
        self.dropout_edge = SparseDropout(dropout_edge)

    def forward(self, x, adj):
        adj = self.dropout_edge(adj)
        for layer in self.conv_layers[: -1]:
            x = layer(x, adj)
            x = self.dropout_node(F.relu(x))
        x = self.conv_layers[-1](x, adj)
        return x





