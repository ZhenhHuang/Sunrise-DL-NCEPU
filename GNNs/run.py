import torch
import numpy as np
import os
import random
import argparse


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='experiences for GNNs')

# data
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of datasets')
parser.add_argument('--data', type=str, default='cora', help='name of datasets')

# model
parser.add_argument('--model', type=str, default='gcn', help='model for experience')
parser.add_argument('--in_features', type=int, default=1433, help='dimension of features')
parser.add_argument('--hidden_features', type=int, default=16, help='dimension of hidden features')
parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--activation', type=str, default='relu', help='activate function')

# train
parser.add_argument('--epochs', type=int, default=200, help='epochs for training')
parser.add_argument('--lr', type=float, default=1e-2, help='learning_rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='rate of weight_decay')
parser.add_argument('--model_path', type=str, default='gcn.pt', help='path for saving models')
parser.add_argument('--patience', type=int, default=3, help='patience of early stopping')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

args = parser.parse_args()


if __name__ == '__main__':
    from GNNs.exp import Exp
    exp = Exp(args)
    exp.train()