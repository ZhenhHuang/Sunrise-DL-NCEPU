import torch
import numpy as np
import time
from image_classification.data_loader import getLoader
import faiss
from tools import EarlyStopping, adjust_learning_rate


def train(args, model, criterion, optimizer, device, transform=None):
    model.train()
    time_now = time.time()
    epochs = args.epochs
    print('load train')
    train_set, train_loader = getLoader(args, flag='train', transform=transform)
    print('load valid')
    val_set, val_loader = getLoader(args, flag='val', transform=transform)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for iters in range(epochs):
        losses = []
        count = 0
        epoch_time = time.time()
        for i, (data, label) in enumerate(train_loader):
            count += 1
            data = data.float().to(device)
            label = label.long().to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                print(f"iter {i}, loss: {loss.item()}")
                speed = (time.time() - time_now) / count
                left_time = speed * ((epochs - iters) * len(train_loader) - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                count = 0
                time_now = time.time()
        print("Epoch: {} cost time: {}".format(
            iters + 1, time.time() - epoch_time))
        val_loss = val(val_loader, model, criterion, device)
        print(f"Epoch {iters+1}, train_loss: {np.mean(losses)}, val_loss: {val_loss}")
        early_stopping(val_loss, model, args.save_id)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def val(test_loader, model, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.float().to(device)
            label = label.long().to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.detach().cpu().numpy())
    model.train()
    return np.mean(losses)


def test(args, model, device, transform=None):
    results = []
    trues = []
    preds = []
    print('load test')
    test_set, test_loader = getLoader(args, flag='test', transform=transform)
    state_dict_path = f'./checkpoints/{args.save_id if args.exp_type != "distil" else args.st_path}'
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.float().to(device)
            output = model(data)
            output = torch.softmax(output, dim=-1)
            result = torch.argmax(output).cpu().numpy()
            label = label.cpu().numpy()
            results.append((label == result).sum())
            trues.append(label.item())
            preds.append(result.item())
    np.save(f'{args.results_path}/trues.npy', np.array(trues))
    np.save(f'{args.results_path}/preds.npy', np.array(preds))
    acc = np.sum(results) / len(test_set)
    return acc


def triple_train(args, model, criterion, optimizer, device, transform=None):
    model.train()
    time_now = time.time()
    epochs = args.epochs
    print('load train')
    train_set, train_loader = getLoader(args, flag='train', transform=transform)
    print('load valid')
    val_set, val_loader = getLoader(args, flag='val', transform=transform)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for iters in range(epochs):
        losses = []
        count = 0
        epoch_time = time.time()
        for i, (data1, data2, data3) in enumerate(train_loader):
            count += 1
            x1, y1 = data1
            x2, y2 = data2
            x3, y3 = data3
            x1, y1 = x1.float().to(device), y1.long().to(device)
            x2, y2 = x2.float().to(device), y2.long().to(device)
            x3, y3 = x3.float().to(device), y3.long().to(device)
            output1, output2, output3 = model(x1), model(x2), model(x3)
            loss = criterion(output1, output2, output3, y1, y2, y3)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                print(f"iter {i}, loss: {loss.item()}")
                speed = (time.time() - time_now) / count
                left_time = speed * \
                            ((epochs - iters) * len(train_loader) - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                count = 0
                time_now = time.time()
        print("Epoch: {} cost time: {}".format(
            iters + 1, time.time() - epoch_time))
        val_loss = triple_val(val_loader, model, criterion, device)
        print(f"Epoch {iters + 1}, train_loss: {np.mean(losses)}, val_loss: {val_loss}")
        early_stopping(val_loss, model, args.save_id)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def triple_val(test_loader, model, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (data1, data2, data3) in enumerate(test_loader):
            x1, y1 = data1
            x2, y2 = data2
            x3, y3 = data3
            x1, y1 = x1.float().to(device), y1.long().to(device)
            x2, y2 = x2.float().to(device), y2.long().to(device)
            x3, y3 = x3.float().to(device), y3.long().to(device)
            output1, output2, output3 = model(x1), model(x2), model(x3)
            loss = criterion(output1, output2, output3, y1, y2, y3)
            losses.append(loss.detach().cpu().numpy())
    model.train()
    return np.mean(losses)


def triple_test(args, model, device, index: faiss.IndexFlatL2, transform=None):
    state_dict = torch.load(f'./checkpoints/{args.save_id}')
    model.load_state_dict(state_dict)
    model.eval()
    print('load train')
    train_set, train_loader = getLoader(args, flag='train', transform=transform)
    print('load test')
    test_set, test_loader = getLoader(args, flag='val', transform=transform)
    targets = []
    results = []
    trues = []
    preds = []
    with torch.no_grad():
        for i, (data, label) in enumerate(train_loader):
            data = data.float().to(device)
            output = model(data)
            output = output.cpu().numpy()
            index.add(output)
            targets.append(label)
        targets = torch.concat(targets)
        for i, (data, label) in enumerate(test_loader):
            time_now = time.time()
            data = data.float().to(device)
            output = model(data)
            output = output.cpu().numpy()
            _, I = index.search(output, 5)
            result = targets[I[0][0]]
            if i % 200 == 0:
                cost_time = time.time() - time_now
                print(f"cost time: {cost_time} s, average: {cost_time / 200} s")
                time_now = time.time()
            results.append((label == result).sum())
            trues.append(label.item())
            preds.append(result.item())

    acc = np.sum(results) / len(test_set)
    trues = np.array(trues)
    preds = np.array(preds)
    f = open("result_triple.txt", 'w', encoding='utf-8')
    f.write(f"total\t{acc * 100}%")
    for k, v in test_set.map_dict.items():
        trues_k = trues[trues == k]
        preds_k = preds[trues == k]
        acc_k = np.sum((trues_k == preds_k).astype(int))
        f.write(f"\n{v}\t{acc_k / len(trues_k) * 100}%")
    f.close()
    return acc


def distil_train(args, st_model, teacher, criterion, optimizer, device, transform=None):
    print('-----------------load teacher model-----------------')
    state_dict = torch.load(f'./checkpoints/{args.teacher_path}')
    teacher.load_state_dict(state_dict)
    for name, param in teacher.named_parameters():
        param.requires_grad = False
    for name, param in teacher.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
    print('load train')
    train_set, train_loader = getLoader(args, flag='train', transform=transform)
    print('load valid')
    val_set, val_loader = getLoader(args, flag='val', transform=transform)

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
        print(f"Epoch {iters+1}, train_loss: {np.mean(losses)}, valid_loss: {val_loss}")
        early_stopping(val_loss, st_model, args.st_path)
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




