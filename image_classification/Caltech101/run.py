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

parser = argparse.ArgumentParser(description='')

# load data
parser.add_argument('--root_path', type=str, default='../data/data146107', help='root_path of data')
parser.add_argument('--data_path', type=str, default='dataset', help='path of data')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--load', action='store_false', help='load .npy file for quick load')
parser.add_argument('--size', type=list, default=[224, 224],
                    help='input image size')

# train
parser.add_argument('--epochs', type=int, default=30, help='epochs of train')
parser.add_argument('--patience', type=int, default=3,
                    help='early stopping patience')
parser.add_argument('--teacher_path', type=str, default='model.pt')
parser.add_argument('--T', type=int, default=2, help='temperature of distil')
parser.add_argument('--alpha', type=float, default=0.9, help='rate of two losses')
# model
parser.add_argument('--save_id', type=str, default='model', help='name for save checkpoints')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--learning_rate', type=float, default=1e-4)

#GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multile gpus')


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print(args.use_gpu)


if __name__ == '__main__':
    import torch.nn as nn
    import torch.optim as optim
    from utils import train, test, distil_train
    from model import DPSENet, DistilLoss
    from torchvision.models.resnet import resnet50
    print(args)
    device = torch.device('cuda:0') if args.use_gpu else torch.device('cpu')
    teacher = resnet50(pretrained=True).to(device)
    teacher.fc = nn.Linear(2048, 102).to(device)
    
    st_model = DPSENet()
    
    # criterion = nn.CrossEntropyLoss()
    criterion = DistilLoss(args.T, args.alpha)
    learning_rate = 1e-4
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    #print('-----------------train start-----------------')
    #state_dict = torch.load(f'{args.save_id}.pt')
    #model.load_state_dict(state_dict)
    #train(args, model, criterion=criterion, optimizer=optimizer, device=device)
    print('-----------------fine tuning start-----------------')
    distil_train(args, st_model, teacher, criterion, optimizer, device=device)
    print('-----------------test start-----------------')
    #test(args, model, device)
    torch.cuda.empty_cache()













