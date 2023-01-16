import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Attentive, GraphConvolution
from utils import k_nearest_neighbors, cal_similarity_graph, graph_top_K, apply_non_linearity


class FGP_learner(nn.Module):
    def __init__(self, features, k_neighbours, knn_metric: str, slope, alpha):
        super(FGP_learner, self).__init__()
        self.k_neighbours = k_neighbours
        self.knn_metric = knn_metric
        self.adj = nn.Parameter(torch.tensor(k_nearest_neighbors(features, k_neighbours, knn_metric)))
        self.slope = slope
        self.alpha = alpha

    def forward(self, x):
        adj = F.elu(self.slope * (self.adj - 1), alpha=self.alpha) + self.alpha
        return adj


class ATT_learner(nn.Module):
    def __init__(self, n_layers, in_features, k_neighbours, knn_metric, activation, slope, alpha):
        """slope, alpha are for ELU"""
        super(ATT_learner, self).__init__()
        self.att_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.att_layers.append(Attentive(in_features))
        self.k_neighbours = k_neighbours
        self.knn_metric = knn_metric
        self.activation = F.relu if activation == 'relu' else F.tanh
        self.slope = slope
        self.alpha = alpha

    def forward(self, x):
        for layer in self.att_layers[:-1]:
            x = self.activation(layer(x))
        x = self.att_layers[-1](x)
        x = F.normalize(x, dim=-1, p=2)
        x = cal_similarity_graph(x)
        x = graph_top_K(x, self.k_neighbours + 1)
        x = apply_non_linearity(x, 'relu', self.slope, self.alpha)
        return x


class MLP_learner(nn.Module):
    def __init__(self, n_layers, in_features, k_neighbours, knn_metric, activation, slope, alpha):
        super(MLP_learner, self).__init__()
        self.mlp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mlp_layers.append(nn.Linear(in_features, in_features))
        self.init_parameters(in_features)
        self.k_neighbours = k_neighbours
        self.knn_metric = knn_metric
        self.activation = F.relu if activation == 'relu' else F.tanh
        self.slope = slope
        self.alpha = alpha

    def init_parameters(self, in_features):
        for layer in self.mlp_layers:
            layer.weight = nn.Parameter(torch.eye(in_features))

    def forward(self, x):
        for layer in self.mlp_layers[:-1]:
            x = self.activation(layer(x))
        x = self.mlp_layers[-1](x)
        x = F.normalize(x, dim=-1, p=2)
        x = cal_similarity_graph(x)
        x = graph_top_K(x, self.k_neighbours + 1)
        x = apply_non_linearity(x, 'relu', self.slope, self.alpha)
        return x


class GNN_learner(nn.Module):
    def __init__(self, n_layers, in_features, k_neighbours,
                 knn_metric, activation, slope, alpha, adj):
        super(GNN_learner, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GraphConvolution(in_features, in_features))
        self.init_parameters(in_features)
        self.k_neighbours = k_neighbours
        self.knn_metric = knn_metric
        self.activation = F.relu if activation == 'relu' else F.tanh
        self.slope = slope
        self.alpha = alpha
        self.adj = adj

    def init_parameters(self, in_features):
        for layer in self.layers:
            for net in layer.modules():
                if isinstance(net, nn.Linear):
                    net.weight = nn.Parameter(torch.eye(in_features))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, self.adj))
        x = self.layers[-1](x, self.adj)
        x = F.normalize(x, dim=-1, p=2)
        x = cal_similarity_graph(x)
        x = graph_top_K(x, self.k_neighbours + 1)
        x = apply_non_linearity(x, 'relu', self.slope, self.alpha)
        return x














