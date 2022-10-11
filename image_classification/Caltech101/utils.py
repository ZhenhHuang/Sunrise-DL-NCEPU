import torch
import numpy as np
import time
from data_loader import ImageData, DataLoader


def train(args, model, criterion, optimizer, device):
    model.train()
    time_now = time.time()
    epochs = args.epochs
    print('load train')
    train_set = ImageData(args.root_path, args.data_path, flag='train', size=args.size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print('load val')    
    val_set = ImageData(args.root_path, args.data_path, flag='val', size=args.size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print('load test')
    test_set = ImageData(args.root_path, args.data_path, flag='test', size=args.size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(len(train_set), len(val_set), len(test_set))
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for iters in range(epochs):
        losses = []
        count = 0
        epoch_time = time.time()
        for i, (data, label) in enumerate(train_loader):
            count += 1
            data = data.to(device).float()
            label = label.long().to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"iter {i}, loss: {loss.item()}")
                speed = (time.time() - time_now) / count
                left_time = speed * \
                            ((epochs - iters) * len(train_loader) - i)
                print(
                    '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                count = 0
                time_now = time.time()
        print("Epoch: {} cost time: {}".format(
            iters + 1, time.time() - epoch_time))
        val_loss = val(val_loader, model, criterion, device)
        test_loss = val(test_loader, model, criterion, device)
        print(f"Epoch {iters+1}, train_loss: {np.mean(losses)}, val_loss: {val_loss}, test_loss: {test_loss}")
        early_stopping(val_loss, model, args.save_id)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        adjust_learning_rate(optimizer, iters + 1, args) 


def val(test_loader, model, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device).float()
            label = label.long().to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.detach().cpu().numpy())
    model.train()
    return np.mean(losses)


def test(args, model, device):
    results = []
    print('load pred')
    pred_set = ImageData(flag='pred', size=args.size)
    pred_loader = DataLoader(pred_set, batch_size=1, shuffle=False, drop_last=False)
    state_dict = torch.load(f'{args.save_id}.pt')
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(pred_loader):
            data = data.to(device).float()
            output = model(data)
            output = torch.softmax(output, dim=-1)
            result = torch.argmax(output).cpu().numpy()
            results.append(result)
    f = open("result.txt", 'a')
    for i, result in enumerate(results):
        f.write(f"{pred_set.data_names[i]}\t{result}")
        if i < len(results)-1:
            f.write("\n")
    f.close()


def distil_train(args, st_model, teacher, criterion, optimizer, device):
    print('--------------load teacher model---------------')
    state_dict = torch.load(f'{args.teacher_path}')
    teacher.load_state_dict(state_dict)
    for name, param in teacher.named_parameters():
        param.requires_grad = False
    for name, param in teacher.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
    print('load train')
    train_set = ImageData(args.root_path, args.data_path, flag='train', size=args.size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print('load val')    
    val_set = ImageData(args.root_path, args.data_path, flag='val', size=args.size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print('load test')
    test_set = ImageData(args.root_path, args.data_path, flag='test', size=args.size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print('-------train student model---------')
    st_model.train()
    time_now = time.time()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    epochs = args.epochs
    for iters in range(epochs):
        losses = []
        count = 0
        epoch_time = time.time()
        for i, (data, label) in enumerate(train_loader):
            count += 1
            data = data.float().to(device)
            label = label.long().to(device)
            output1 = st_model(data)
            output2 = teacher(data)
            loss = criterion(output1, output2, label)
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
        print("Epoch: {} cost time: {}".format(iters + 1, time.time() - epoch_time))
        val_loss = distil_val(val_loader, st_model, teacher, criterion, device)
        test_loss = distil_val(test_loader, st_model, teacher, criterion, device)
        print(f"Epoch {iters+1}, train_loss: {np.mean(losses)}, valid_loss: {val_loss}, test_loss: {test_loss}")
        early_stopping(val_loss, st_model, args.save_id)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        adjust_learning_rate(optimizer, iters + 1, args) 


def distil_val(test_loader, st_model, teacher, criterion, device):
    st_model.eval()
    losses = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.float().to(device)
            label = label.long().to(device)
            output1 = st_model(data)
            output2 = teacher(data)
            loss = criterion(output1, output2, label)
            losses.append(loss.detach().cpu().numpy())
    st_model.train()
    return np.mean(losses)


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
        torch.save(model.state_dict(), f"{path}.pt")
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch // 1)))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
