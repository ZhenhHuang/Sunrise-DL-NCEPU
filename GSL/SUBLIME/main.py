import torch
import numpy as np
import os
import random
import argparse
from exp import Exp


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='SUBLIME')

# Experiment settings
parser.add_argument('--sparse', action='store_true', help='whether the input is sparse matrix form')
parser.add_argument('--gsl_mode', type=str, default='structure_inference',
                    choices=['structure_inference', 'structure_refinement'])
parser.add_argument('--downstream_task', type=str, default='classification',
                    choices=['classification', 'clustering'])
parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora'])
parser.add_argument('--root_path', type=str, default='../datasets')
parser.add_argument('--learner_type', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
parser.add_argument('--eval_freq', type=int, default=5)

# Graph Contrastive Learning Module
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--hidden_features', type=int, default=512)
parser.add_argument('--embed_features', type=int, default=64, help='dimensions of graph embedding')
parser.add_argument('--proj_features', type=int, default=64, help="out dimensions of encoder's projection")
parser.add_argument('--n_layers_gcl', type=int, default=2)
parser.add_argument('--dropout_node', type=float, default=0.5)
parser.add_argument('--dropout_edge', type=float, default=0.5)
parser.add_argument('--contract_batch_size', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--w_decay', type=float, default=0.0)

# Graph Structure Learning Module
parser.add_argument('--k_neighbours', type=int, default=30, help='numbers of K neighbours')
parser.add_argument('--knn_metric', type=str, default='cosine', choices=['cosine', 'minkowski'])
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh'])
parser.add_argument('--alpha', type=float, default=1., help='alpha for elu')
parser.add_argument('--slope', type=float, default=6., help='slope for elu input, elu(slope * (x-1)) + alpha')
parser.add_argument('--n_layers_gsl', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.2, help='temperature of NT-Xent loss')

# data augmentation
parser.add_argument('--maskfeat_prob_anchor', type=float, default=0.2, help='probability of feature mask for anchor view')
parser.add_argument('--maskfeat_prob_learner', type=float, default=0.2, help='probability of feature mask for learner view')

# Evaluation NetWork for Classification
parser.add_argument('--hidden_features_cls', type=int, default=32)
parser.add_argument('--dropout_node_cls', type=float, default=0.5)
parser.add_argument('--dropout_edge_cls', type=float, default=0.25)
parser.add_argument('--n_layers_cls', type=int, default=2)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--w_decay_cls', type=float, default=1e-4)
parser.add_argument('--epochs_cls', type=int, default=200)
parser.add_argument('--patience_cls', type=int, default=15)
parser.add_argument('--save_path_cls', type=str, default='./checkpoints/cls.pth')

# Evaluation NetWork for Clustering
parser.add_argument('--n_cluster_trials', type=int, default=5)

# Structure Bootstrapping
parser.add_argument('--tau', type=float, default=1.)
parser.add_argument('--iterations', type=int, default=0)

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

configs = parser.parse_args()

print(configs)

exp = Exp(configs)
exp.train()

