import numpy as np
import pandas as pd
import os
import torch


class Cora:
    def __init__(self, root_path="./datasets", **kwargs):
        self.train_len = 140
        self.val_len = 500
        self.test_len = 1000
        self.map_dict = {
            "Case_Based": 0,
            "Genetic_Algorithms": 1,
            "Neural_Networks": 2,
            "Probabilistic_Methods": 3,
            "Reinforcement_Learning": 4,
            "Rule_Learning": 5,
            "Theory": 6
        }
        self.n_classes = 7
        self.__read_data__(f"{root_path}/cora")

    def __call__(self, *args, **kwargs):
        mask = (self.mask_train, self.mask_val, self.mask_test)
        return self.data, self.num_node_features, self.label, self.edge_index, mask, self.n_classes

    def __read_data__(self, path):
        raw_data = pd.read_csv(f"{path}/cora.content", sep='\t', header=None)
        data = raw_data.values[:, 1: -1].astype(np.float)
        data = data / data.sum(-1)[:, None]
        self.data = torch.tensor(data).float()
        idx = raw_data.values[:, 0]
        idx_map = {k: v for v, k in enumerate(idx)}
        self.num_node_features = data.shape[-1]
        label = [self.map_dict[k] for k in raw_data.values[:, -1]]
        self.label = torch.tensor(label).long()
        cites = pd.read_csv(f"{path}/cora.cites", sep='\t', header=None)
        edges = torch.eye(len(label))
        for i in range(len(cites)):
            id1, id2 = idx_map[cites.iloc[i, 0]], idx_map[cites.iloc[i, 1]]
            edges[id1, id2] = 1.
            edges[id2, id1] = 1.
        self.edge_index = edges.float()
        self.mask_train = range(140)
        self.mask_val = range(140, 640)
        self.mask_test = range(1708, 2708)

    def __len__(self):
        return 1
