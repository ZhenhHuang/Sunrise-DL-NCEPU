import torch
import numpy as np
import random
import argparse


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser(description="object detection")

# dataset
parser.add_argument('--data', type=str, default='VOC', help='name of dataset')
parser.add_argument('--root_path', type=str, default="C:/Users/98311/Downloads", help='root path of dataset')
parser.add_argument('--year', type=int, default=2007, help='year of VOC dataset')
parser.add_argument('--json_file', type=str, default='./data/pascal_classes_2007.json',
                    help="json file of class mapping")
parser.add_argument('--S', type=int, default=7, help='grids of image to split')
parser.add_argument('--B', type=int, default=2, help='boxes number for each grid')

# dataloader
parser.add_argument('--batch_size', type=int, default=8, help='batch size of dataloader')
parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')
parser.add_argument('--size', type=list, default=[448, 448], help='[width, height]')
parser.add_argument('--threshold', type=float, default=0.1)

# model
parser.add_argument('--dropout', type=float, default=0.5, help='dropout after the first connected layer')
parser.add_argument('--num_classes', type=int, default=20)
parser.add_argument('--w_coord', type=float, default=5.)
parser.add_argument('--w_noobj', type=float, default=.5)
parser.add_argument('--backbone', type=str, default='resnet50', help='backbone of model: [darknet, resnet50, vgg11, vgg16, vgg16_bn]')

# train
parser.add_argument('--epochs', type=int, default=135, help='train epochs')
parser.add_argument('--verbose', type=int, default=20, help='verbose frequency of train steps')
parser.add_argument('--patience', type=int, default=3, help='patience for early stopping')
parser.add_argument('--delta', type=float, default=0., help='error of two valid loss')

# optimizer
parser.add_argument('--optim', type=str, default='SGD', help='[SGD, Adam]')
parser.add_argument('--loss', type=str, default='yolo', help='Loss type: [yolo, mse, cross_entropy]')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--lradj', type=str, default='type1', help='type of learning rate adjustment')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

# save information
parser.add_argument('--model_path', type=str, default='res50.pt', help='path for saving model')
parser.add_argument('--result_path', type=str, default='results', help='path for saving results')

args = parser.parse_args()


if __name__ == '__main__':
    from exp import train
    from models.yolo_v1 import YOLO_V1
    from detect import detect
    from eval_metrics.voc_eval import VOCMetric
    print(args)
    device = torch.device('cuda:0') if torch.cuda.is_available() and args.use_gpu else torch.device('cpu')
    print(device)
    print(f'backbone: {args.backbone}')
    model = YOLO_V1(args.backbone, 7, 2, 20).to(device)
    # train(args, model, device)
    detect(args, model, device)
    eval = VOCMetric()
    map = eval.evaluate()
    torch.cuda.empty_cache()

