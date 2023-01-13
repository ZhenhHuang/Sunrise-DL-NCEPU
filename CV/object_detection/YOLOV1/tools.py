import torch
import numpy as np
import torch.optim as optim
from loss import YOLOLoss
import json
import matplotlib.pyplot as plt


def choose_optim(model, args):
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def choose_loss(args):
    if args.loss == 'yolo':
        return YOLOLoss(**vars(args))
    elif args.loss == 'mse':
        return torch.nn.MSELoss()
    elif args.loss == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f"./checkpoints/{path}")
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {
            0: 1e-3, 75: 1e-4, 105: 1e-5
        }
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (epoch // 2))}
    else:
        return 0
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        return 1
    return 0


def visualize(img, boxes, classes, probs, k):
    color = [[0, 0, 0],
             [128, 0, 0],
             [0, 128, 0],
             [128, 128, 0],
             [0, 0, 128],
             [128, 0, 128],
             [0, 128, 128],
             [128, 128, 128],
             [64, 0, 0],
             [192, 0, 0],
             [64, 128, 0],
             [192, 128, 0],
             [64, 0, 128],
             [192, 0, 128],
             [64, 128, 128],
             [192, 128, 128],
             [0, 64, 0],
             [128, 64, 0],
             [0, 192, 0],
             [128, 192, 0],
             [0, 64, 128]]
    color = np.array(color) / 255.
    f = open('./data/pascal_classes_2007.json')
    map_dict = json.load(f)
    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        index = map_dict[classes[i]]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=tuple(color[index + 1]),
                             label=f"{classes[i]}:{probs[i]}")
        ax.text(xmin, ymin - 5, f"{classes[i]}:{probs[i]}", fontsize=5, color='white',
                bbox=dict(boxstyle='round,pad=0.5', fc=tuple(color[index + 1]), ec=tuple(color[index + 1]), lw=2, alpha=0.7))
        ax.add_patch(rect)
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig(f"./pics/{k}.pdf")
    # plt.show()
