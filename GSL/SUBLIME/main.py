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


# Evaluation NetWork for Classification
parser.add_argument('--hidden_features_cls', type=int, default=32)
parser.add_argument('--dropout_node', type=float, default=0.5)
parser.add_argument('--dropout_edge', type=float, default=0.25)
parser.add_argument('--n_layers', type=int, default=2)