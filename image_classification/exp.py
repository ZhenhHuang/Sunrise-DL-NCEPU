import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from image_classification.utils import train, test, triple_train, triple_test, distil_train
import faiss


class Exp:
    def __init__(self, args, model, criterion, optimizer, teacher_model=None, transform=None):
        self.args = args
        self.device = torch.device('cuda:0') if self.args.use_gpu else torch.device('cpu')
        self.exp_type = args.exp_type
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.teacher_model = teacher_model
        self.transform = transform or transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])

    def train(self):
        print('-----------------train start-----------------')
        print(f'train type: {self.exp_type}')
        if self.exp_type == 'cls':
            train(self.args, self.model, criterion=self.criterion,
                  optimizer=self.optimizer, device=self.device, transform=self.transform)
        elif self.exp_type == 'distl':
            train(self.args, self.model, nn.CrossEntropyLoss(), self.optimizer, self.device, transform=self.transform)
            triple_train(self.args, self.model, self.criterion, self.optimizer, self.device, transform=self.transform)
        else:
            if self.teacher_model is not None:
                distil_train(self.args, self.model, self.teacher_model, self.criterion, self.optimizer, device=self.device, transform=self.transform)
            else:
                raise NotImplementedError

    def test(self):
        print('-----------------test start-----------------')
        print(f'test type: {self.exp_type}')
        if self.exp_type == 'cls':
            acc = test(self.args, self.model, self.device, transform=self.transform)
        elif self.exp_type == 'distl':
            index = faiss.IndexFlatL2(self.args.index_dim)
            acc = triple_test(self.args, self.model, self.device, index, transform=self.transform)
        else:
            if self.teacher_model is not None:
                acc = test(self.args, self.model, self.device, transform=self.transform)
            else:
                raise NotImplementedError
        print(acc)
        return acc


