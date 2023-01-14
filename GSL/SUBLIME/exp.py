import torch
import numpy as np
import torch.nn.functional as F
from models import GCN, GCL
from graph_learners import FGP_learner, ATT_learner, MLP_learner, GNN_learner


class Exp:
    def __init__(self):
        pass