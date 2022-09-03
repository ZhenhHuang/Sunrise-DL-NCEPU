import torch
from torch.utils.data import DataLoader
import numpy as np
import time


def getDataLoader(dataset, batch_size=32, flag='train'):
    if flag == 'train':
        batch_size = batch_size
        shuffle = True
        drop_last = True
    elif flag == 'pred':
        batch_size = 1
        shuffle = False
        drop_last = False
    else:
        batch_size = 1
        shuffle = False
        drop_last = True

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


def train(train_loader, val_loader, test_loader, model, epochs, criterion, optimizer, device, patience=3):
    model.train()
    time_now = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for iters in range(epochs):
        losses = []
        count = 0
        epoch_time = time.time()
        for i, (data, label) in enumerate(train_loader):
            count += 1
            data = data.to(torch.float32).to(device)
            label = label.long().to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"iter {i}, loss: {loss.item()}")
                speed = (time.time() - time_now) / count
                left_time = speed * ((epochs - iters) * len(train_loader) - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                count = 0
                time_now = time.time()
        print("Epoch: {} cost time: {}".format(
            iters + 1, time.time() - epoch_time))
        val_loss = val(val_loader, model, criterion, device)
        test_loss = val(test_loader, model, criterion, device)
        print(f"Epoch {iters+1}, train_loss: {np.mean(losses)}, valid_loss: {val_loss}, test_loss: {test_loss}")
        early_stopping(val_loss, model, './')
        if early_stopping.early_stop:
            print("Early stopping")
            break


def val(test_loader, model, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(torch.float32).to(device)
            label = label.long().to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.detach().cpu())
    model.train()
    return np.mean(losses)


def test(test_set, loader, model, device):
    results = []
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(loader):
            data = data.to(torch.float32).to(device)
            label = label.long().to(device)
            output = model(data)
            output = torch.softmax(output, dim=-1)
            result = torch.argmax(output).item()
            results.append(result)
    f = open("result.txt", 'a')
    for i, result in enumerate(results):
        f.write(f"{test_set.data_names[i]}\t{result}")
        if i < len(results)-1:
            f.write("\n")
    f.close()


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
        torch.save(model.state_dict(), f"{path}/model.pt")
        self.val_loss_min = val_loss