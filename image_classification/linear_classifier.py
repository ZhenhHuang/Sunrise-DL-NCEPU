import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.load_datasets import load_CIFAR, load_MNIST
from models.linear_models import LinearModel
import torchvision.transforms as transforms
import numpy as np


def getDataLoader(dataset, batch_size=32, train: bool = True):
    if train:
        shuffle = True
        drop_last = False
    else:
        shuffle = False
        drop_last = False
        batch_size = 1

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


def train(train_loader, test_loader, model, epochs, criterion, optimizer, device):
    model.train()
    for iters in range(epochs):
        losses = []
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                print(f"iter {i}, loss: {loss.item()}")
        print(f"epoch {iters}, train_loss: {np.mean(losses)}")
        test(test_loader, model, criterion, device)


def test(loader, model, criterion, device):
    model.eval()
    with torch.no_grad():
        losses = []
        acc = 0
        for i, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.item())
            result = torch.argmax(output, dim=-1)
            acc += (result == label).sum()
        print(f"test_loss: {np.mean(losses)}, accuracy: {acc/len(loader)*100: .2f}%, num={acc}")


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset, testset = load_CIFAR(root_path="../datasets/CIFAR", transform=transform)
    # trainset, testset = load_MNIST(root_path="../datasets", transform=transform, download=False)
    train_loader = getDataLoader(trainset)
    test_loader = getDataLoader(testset, train=False)
    model = LinearModel(num_hidden=32*32*3, classes=10).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)
    train(train_loader, test_loader, model, epochs=10, criterion=criterion, optimizer=optimizer, device=device)
    test(test_loader, model, criterion, device)
