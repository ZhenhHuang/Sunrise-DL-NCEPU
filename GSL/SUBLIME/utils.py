import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn import metrics
from munkres import Munkres


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


class cluster_metrics:
    def __init__(self, trues, predicts):
        self.trues = trues
        self.predicts = predicts

    def clusterAcc(self):
        l1 = list(set(self.trues))
        l2 = list(set(self.predicts))
        num1 = len(l1)
        num2 = len(l2)
        if num1 != num2:
            raise Exception("number of classes not equal")

        """compute the cost of allocating c1 in L1 to c2 in L2"""
        cost = np.zeros((num1, num2), dtype=int)
        for i, c1 in enumerate(l1):
            maps = np.where(self.trues == c1)[0]
            for j, c2 in enumerate(l2):
                maps_d = [i1 for i1 in maps if self.predicts[i1] == c2]
                cost[i, j] = len(maps_d)

        mks = Munkres()
        index = mks.compute(-cost)
        new_predicts = np.zeros(len(self.predicts))
        for i, c in enumerate(l1):
            c2 = l2[index[i][1]]
            allocate_index = np.where(self.predicts == c2)[0]
            new_predicts[allocate_index] = c

        acc = metrics.accuracy_score(self.trues, new_predicts)
        f1_macro = metrics.f1_score(self.trues, new_predicts, average='macro')
        precision_macro = metrics.precision_score(self.trues, new_predicts, average='macro')
        recall_macro = metrics.recall_score(self.trues, new_predicts, average='macro')
        f1_micro = metrics.f1_score(self.trues, new_predicts, average='micro')
        precision_micro = metrics.precision_score(self.trues, new_predicts, average='micro')
        recall_micro = metrics.recall_score(self.trues, new_predicts, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluateFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.trues, self.predicts)
        adjscore = metrics.adjusted_rand_score(self.trues, self.predicts)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusterAcc()
        return acc, nmi, f1_macro, adjscore


# class cluster_metrics:
#     def __init__(self, true_label, predict_label):
#         self.true_label = true_label
#         self.pred_label = predict_label
#
#     def clusteringAcc(self):
#         # best mapping between true_label and predict label
#         l1 = list(set(self.true_label))
#         numclass1 = len(l1)
#
#         l2 = list(set(self.pred_label))
#         numclass2 = len(l2)
#         if numclass1 != numclass2:
#             print('Class Not equal, Error!!!!')
#             return 0, 0, 0, 0, 0, 0, 0
#
#         cost = np.zeros((numclass1, numclass2), dtype=int)
#         for i, c1 in enumerate(l1):
#             mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
#             for j, c2 in enumerate(l2):
#                 mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]
#
#                 cost[i][j] = len(mps_d)
#
#         # match two clustering results by Munkres algorithm
#         m = Munkres()
#         cost = cost.__neg__().tolist()
#
#         indexes = m.compute(cost)
#
#         # get the match results
#         new_predict = np.zeros(len(self.pred_label))
#         for i, c in enumerate(l1):
#             # correponding label in l2:
#             c2 = l2[indexes[i][1]]
#
#             # ai is the index with label==c2 in the pred_label list
#             ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
#             new_predict[ai] = c
#
#         acc = metrics.accuracy_score(self.true_label, new_predict)
#         f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
#         precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
#         recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
#         f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
#         precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
#         recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
#         return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro
#
#     def evaluateFromLabel(self, print_results=True):
#         nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
#         adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
#         acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()
#
#         if print_results:
#             print('ACC={:.4f}, f1_macro={:.4f}, precision_macro={:.4f}, recall_macro={:.4f}, f1_micro={:.4f}, '
#                   .format(acc, f1_macro, precision_macro, recall_macro, f1_micro) +
#                   'precision_micro={:.4f}, recall_micro={:.4f}, NMI={:.4f}, ADJ_RAND_SCORE={:.4f}'
#                   .format(precision_micro, recall_micro, nmi, adjscore))
#
#         return acc, nmi, f1_macro, adjscore







