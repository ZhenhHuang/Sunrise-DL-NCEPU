import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import numpy as np


def k_nearest_neighbors(x, k_neighbours, metric):
    adj = kneighbors_graph(x, k_neighbours, metric=metric)
    adj = adj.toarray()
    return adj


def cal_similarity_graph(node):
    return node @ node.transpose(-1, 1)


def graph_top_K(dense_adj, k):
    assert k < dense_adj.shape[-1]
    _, indices = dense_adj.topk(k=k, dim=-1)
    mask = torch.zeros(dense_adj.shape).bool().to(dense_adj.device)
    mask[torch.arange(dense_adj.shape[0])[:, None], indices] = True
    sparse_adj = torch.masked_fill(dense_adj, mask, value=0.)
    return sparse_adj


def apply_non_linearity(x, non_linear_func, slope, alpha):
    if non_linear_func == 'elu':
        return F.elu(slope(x - 1), alpha=alpha) + alpha
    elif non_linear_func == 'relu':
        return F.relu(x)
    elif non_linear_func == 'none':
        return x
    else:
        raise NotImplementedError('the non_linear_function is not implemented')