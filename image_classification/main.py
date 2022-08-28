import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from datasets.load_datasets import load_CIFAR, load_MNIST
from models.linear_models import LinearModel
from models.conv_models import ConvNet, ResNet18
import torchvision.transforms as transforms
import numpy as np
from image_classification.utils import getDataLoader, train, test


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset, testset = load_CIFAR(root_path="../datasets/CIFAR", transform=transform, download=False)
    # trainset, testset = load_MNIST(root_path="../datasets", transform=transform, download=True)
    train_loader = getDataLoader(trainset)
    test_loader = getDataLoader(testset, train=False)

    # model = LinearModel(num_hidden=28*28, classes=10).to(device)
    # model = ConvNet(in_channel=3).to(device)
    model = ResNet18().to(device)

    criterion = nn.NLLLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    train(train_loader, test_loader, model, epochs=15, criterion=criterion, optimizer=optimizer, device=device)
    test(test_loader, model, criterion, device)

    # weight visualization
    # weight = model.linear.weight.detach().cpu().numpy()
    # print(weight.shape)
    # weight = weight.reshape(-1, 28, 28, 1)
    # N = weight.shape[0]
    # rows = 2
    # for i in range(N):
    #     max_w = weight[i].max(0).max(0)[None, None, :]
    #     min_w = weight[i].min(0).min(0)[None, None, :]
    #     print(max_w.shape)
    #     tmp = (weight[i] - min_w) / (max_w - min_w)
    #     plt.subplot(N // rows, rows, i + 1)
    #     plt.imshow(tmp, cmap='gray')
    # plt.show()



