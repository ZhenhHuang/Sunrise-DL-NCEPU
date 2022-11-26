import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features=64, n_heads=8, alpha=0.2, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)
        self.weights = nn.Parameter(torch.empty(size=(2*out_features // n_heads, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.weights.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [N, D]
        N, D = x.shape
        x = self.linear(x).reshape(N, self.n_heads, -1)     # N, H, F
        attn = torch.einsum('nhf, fi->nhi', x, self.weights.reshape(-1, 2))   # N, H, 2
        attn = attn[:, :, 0].unsqueeze(0) + attn[:, :, 1].unsqueeze(1)  # N_1, N_2, H
        attn = torch.masked_fill(F.leaky_relu(attn, self.alpha), mask=~adj.bool().to_dense().unsqueeze(-1), value=-torch.inf)
        scores = torch.softmax(attn, dim=1)
        out = torch.einsum('ijh, jhf->ihf', self.dropout(scores), x).reshape(N, -1)
        return out


class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, n_heads, num_classes, alpha=0.2, dropout=0.6, **kwargs):
        super(GAT, self).__init__()
        self.attnlayer1 = GraphAttentionLayer(in_features, hidden_features, n_heads[0], alpha)
        self.attnlayer2 = GraphAttentionLayer(hidden_features, num_classes, n_heads=n_heads[1], alpha=alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.attnlayer1(x, adj)
        x = self.dropout(F.elu(x))
        x = self.attnlayer2(x, adj)
        return x

