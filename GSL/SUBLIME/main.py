import torch
import numpy as np
import os
import random
import argparse


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='SUBLIME')


# Graph Contrastive Learning Module
parser.add_argument('--hidden_features', type=int, default=512)
parser.add_argument('--embed_features', type=int, default=64, help='dimensions of graph embedding')
parser.add_argument('--proj_features', type=int, default=64, help="out dimensions of encoder's projection")
parser.add_argument('--n_layers_gcl', type=int, default=2)
parser.add_argument('--dropout_node', type=float, default=0.5)
parser.add_argument('--dropout_edge', type=float, default=0.5)

# Graph Structure Learning Module
parser.add_argument('--k_neighbours', type=int, default=30, help='numbers of K neighbours')
parser.add_argument('--knn_metric', type=str, default='cosine', choices=['cosine', 'minkowski'])
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh'])
parser.add_argument('--alpha', type=float, default=1., help='alpha for elu')
parser.add_argument('--slope', type=float, default=6., help='slope for elu input, elu(slope * (x-1)) + alpha')
parser.add_argument('--n_layers_gsl', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.2, help='temperature of NT-Xent loss')


# Evaluation NetWork for Classification
parser.add_argument('--hidden_features_cls', type=int, default=32)
parser.add_argument('--dropout_node_cls', type=float, default=0.5)
parser.add_argument('--dropout_edge_cls', type=float, default=0.25)
parser.add_argument('--n_layers_cls', type=int, default=2)