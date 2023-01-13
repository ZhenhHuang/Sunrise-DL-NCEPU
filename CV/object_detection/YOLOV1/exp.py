import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from data.data_factory import data_factory
from tools import EarlyStopping, adjust_learning_rate, choose_optim, choose_loss


def train(args, model, device):
    print(f"optimizer: {args.optim}")
    optimizer = choose_optim(model, args)
    print(f"Loss: {args.loss}")
    criterion = choose_loss(args)
    print("----------loading train set-------")
    train_set, train_loader = data_factory(args, flag='train')
    print("----------loading valid set-------")
    valid_set, valid_loader = data_factory(args, flag='val')
    print("----------loading test set--------")
    test_set, test_loader = data_factory(args, flag='test')
    print("-----------train start------------")
    early_stopping = EarlyStopping(args.patience, verbose=True, delta=args.delta)
    model.train()
    for epoch in range(args.epochs):
        train_losses = []
        count = 0
        epoch_time = time.time()
        time_now = time.time()
        for i, (image, target) in enumerate(train_loader):
            count += 1
            image = image.float().to(device)
            target = target.float().to(device)
            output = model(image)
            loss = criterion(output, target)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            if i % args.verbose == 0:
                print(f"iter {i}, loss: {loss.item()}")
                speed = (time.time() - time_now) / count
                left_time = speed * ((args.epochs - epoch) * len(train_loader) - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                count = 0
                time_now = time.time()
        print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
        valid_loss = valid(valid_loader, model, criterion, device)
        test_loss = valid(test_loader, model, criterion, device)
        print(f"Epoch {epoch + 1}, train_loss: {np.mean(train_losses)}, val_loss: {valid_loss}, test_loss: {test_loss}")
        early_stopping(valid_loss, model, args.model_path)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        adjust_learning_rate(optimizer, epoch, args)


def valid(valid_loader, model, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (image, target) in enumerate(valid_loader):
            image = image.float().to(device)
            target = target.float().to(device)
            output = model(image)
            loss = criterion(output, target)
            losses.append(loss.detach().cpu().numpy())
    model.train()
    return np.mean(losses)






