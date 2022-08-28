import torch
from torch.utils.data import DataLoader
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
    return losses