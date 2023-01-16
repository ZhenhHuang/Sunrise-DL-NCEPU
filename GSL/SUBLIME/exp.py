import torch
import numpy as np
import torch.nn.functional as F
from models import GCN, GCL
from graph_learners import FGP_learner, ATT_learner, MLP_learner, GNN_learner
from utils import cal_accuracy, get_masked_features


class Exp:
    def __init__(self, configs):
        self.configs = configs

    def cal_cls_loss(self, model, mask, features, labels):
        out = model(features)
        loss = F.cross_entropy(out[mask], labels[mask])
        acc = cal_accuracy(out, features)
        return loss, acc

    def cal_loss_gcl(self, model, graph_learner, features, anchor_adj):
        """anchor view"""
        aug_feats_anchor = get_masked_features(features, self.configs.maskfeat_prob_anchor)
        z_anchor, _ = model(aug_feats_anchor, anchor_adj)

        """learner view"""
        learner_adj = graph_learner(features)
        learner_adj = (learner_adj + learner_adj.t()) / 2.  # symmetrize
        # TODO normalize

        aug_feats_learner = get_masked_features(features, self.configs.maskfeat_prob_learner)
        z_learner, _ = model(aug_feats_learner, learner_adj)

        # TODO consider batch_size
        loss = model.cal_gcl_loss(z_anchor, z_learner, temperature=self.configs.temperature)
        return loss, learner_adj

