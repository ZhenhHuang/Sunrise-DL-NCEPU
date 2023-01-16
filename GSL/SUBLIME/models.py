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


class GraphEncoder(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, embed_features,
                 proj_features, dropout, dropout_edge):
        super(GraphEncoder, self).__init__()
        self.dropout_node = nn.Dropout(dropout)
        self.dropout_adj = SparseDropout(dropout_edge)

        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GraphConvolution(in_features, hidden_features))
        for _ in range(n_layers - 2):
            self.encoder_layers.append(GraphConvolution(hidden_features, hidden_features))
        self.encoder_layers.append(GraphConvolution(hidden_features, embed_features))

        self.projection = nn.Sequential(nn.Linear(embed_features, proj_features),
                                        nn.ReLU(),
                                        nn.Linear(proj_features, proj_features))

    def forward(self, x, adj):
        adj = self.dropout_adj(adj)
        for layer in self.encoder_layers[:-1]:
            x = self.dropout_node(F.relu(layer(x, adj)))
        x = self.encoder_layers[-1](x, adj)
        z = self.projection(x)
        return z, x


class GCL(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, embed_features,
                 proj_features, dropout, dropout_edge):
        super(GCL, self).__init__()
        self.encoder = GraphEncoder(n_layers, in_features, hidden_features, embed_features,
                 proj_features, dropout, dropout_edge)

    def forward(self, x, adj):
        z, embedding = self.encoder(x, adj)
        return z, embedding

    @staticmethod
    def cal_gcl_loss(x1, x2, temperature=0.2, sym=True):
        norm1 = x1.norm(dim=-1)
        norm2 = x2.norm(dim=-1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', norm1, norm2)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix.diag()
        if sym:
            loss_1 = pos_sim / (sim_matrix.sum(dim=-2) - pos_sim)
            loss_2 = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)

            loss_1 = -torch.log(loss_1).mean()
            loss_2 = -torch.log(loss_2).mean()
            loss = (loss_1 + loss_2) / 2.
        else:
            loss = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)
            loss = -torch.log(loss).mean()
        return loss