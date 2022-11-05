import argparse
import os
import torch
import random
import numpy as np
import warnings


warnings.filterwarnings('ignore')

fix_seed = 2022
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='image classification')

# load data
parser.add_argument('--root_path', type=str, default='./datasets', help='root_path of data')
parser.add_argument('--data_path', type=str, default='caltech101', help='path of data')
parser.add_argument('--data', type=str, default='cifar-10', help='names of datasets, from [caltech101, mnist, cifar-10]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--size', type=list, default=[224, 224], help='input image size')
parser.add_argument('--download', action="store_true", help='whether download dataset')

# train
parser.add_argument('--epochs', type=int, default=30, help='epochs of train')
parser.add_argument('--patience', type=int, default=3,
                    help='early stopping patience')
parser.add_argument('--teacher_path', type=str, default='model.pt')
parser.add_argument('--T', type=int, default=2, help='temperature of distil')
parser.add_argument('--alpha', type=float, default=0.9, help='rate of two losses')

# model
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--learning_rate', type=float, default=1e-4)

#GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multile gpus')

# Exp
parser.add_argument('--exp_type', type=str, default='cls', help='training type: [cls, distil, retrievals]')
parser.add_argument('--index_dim', type=int, default=64, help='dimensions of retrival index')
parser.add_argument('--save_id', type=str, default='googlenet.pt', help='name for save checkpoints')
parser.add_argument('--st_path', type=str, default='googlenet.pt', help='name for save student model checkpoints')
parser.add_argument('--results_path', type=str, default='./results', help='path for saving results')


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print(args.use_gpu)


if __name__ == '__main__':
    import torch.nn as nn
    from image_classification.exp import Exp
    from models.googlenet import GoogleNet
    model = GoogleNet(n_classes=10)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())
    exp = Exp(args, model, criterion, optim)
    # exp.train()
    exp.test()
    torch.cuda.empty_cache()













