import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import numpy as np


def k_nearest_neighbors(x, k_neighbours, metric):
    adj = kneighbors_graph(x, k_neighbours, metric=metric)
    adj = adj.toarray().astype(np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def cal_similarity_graph(node):
    return node @ node.transpose(-1, -2)


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


def cal_accuracy(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    correct = (preds == trues).sum()
    return correct / len(trues)


def get_masked_features(features, mask_prob):
    D = features.shape[-1]
    if int(mask_prob * D) == 0:
        return features
    mask = torch.ones(D, device=features.device)
    drop_index = np.random.choice(D, size=int(mask_prob * D), replace=False)
    mask[drop_index] = 0.
    return features * mask


def normalize(adj, mode, sparse=False):
    if sparse:
        # TODO: implement sparse form
        adj = adj.coalesce()
        if mode == 'sym':
            degree_matrix = 1. / (torch.sqrt(torch.sparse.sum(adj, -1)))
            value = degree_matrix[adj.indices()[0]] * degree_matrix[adj.indices()[1]]
        elif mode == 'row':
            degree_matrix = 1. / (torch.sparse.sum(adj, -1))
            value = degree_matrix[adj.indices()[0]]
        else:
            raise NotImplementedError
        return torch.sparse_coo_tensor(adj.indices(), value * adj.values(), adj.shape)
    else:
        if mode == 'sym':
            degree_matrix = 1. / (torch.sqrt(adj.sum(-1)) + 1e-10)
            return degree_matrix[:, None] * adj * degree_matrix[None, :]
        elif mode == 'row':
            degree_matrix = 1. / (adj.sum(-1) + 1e-5)
        else:
            raise NotImplementedError
        return degree_matrix[:, None] * adj


def split_batch(nodes_idx, batch_size):
    count = len(nodes_idx) // batch_size
    split_result = []
    for i in range(count):
        split_result.append(nodes_idx[i*batch_size: (i+1)*batch_size])
    if len(nodes_idx) % batch_size != 0:
        split_result.append(nodes_idx[count * batch_size:])
    return split_result










