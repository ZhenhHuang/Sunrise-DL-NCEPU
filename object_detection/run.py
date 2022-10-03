import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)