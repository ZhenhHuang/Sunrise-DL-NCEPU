import torch
import torch.nn as nn
import torch.optim
import time
from data_loader import Cora
from GNNs.models.gcn import GCN
from GNNs.models.gat import GAT
from GNNs.tools import EarlyStopping, adjust_learning_rate


class Exp:
    def __init__(self, configs):
        self.configs = configs
        self._select_model()
        self._select_device()

    def _select_model(self):
        model_dicts = {
            'gcn': GCN,
            'gat': GAT
        }
        self.model = model_dicts[self.configs.model]

    def _select_device(self):
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def _select_optim(self, model):
        optim = torch.optim.Adam(params=model.parameters(), lr=self.configs.lr,
                                 weight_decay=self.configs.weight_decay)
        return optim

    def _select_criterion(self):
        loss = nn.CrossEntropyLoss()
        return loss

    def _get_data(self):
        data_dict = {
            'cora': Cora
        }
        Data = data_dict[self.configs.data]
        dataset = Data(**vars(self.configs))
        return dataset

    def _calc_acc(self, preds, trues):
        acc = (preds == trues).sum() / len(trues)
        return acc.item()

    def valid(self, model, x, label, edge, mask_val, criterion):
        model.eval()
        with torch.no_grad():
            out = model(x, edge)[mask_val]
            loss = criterion(out, label[mask_val])
        model.train()
        acc = self._calc_acc(out.argmax(-1), label[mask_val])
        return loss.detach().cpu().numpy(), acc

    def train(self):
        data = self._get_data()
        x, label, edge_index, mask_train, mask_val, mask_test = data()
        x = x.to(self.device)
        label = label.to(self.device)
        edge_index = edge_index.to_dense().to(self.device).to_sparse()
        model = self.model(**vars(self.configs)).to(self.device)
        optimizer = self._select_optim(model)
        criterion = self._select_criterion()
        earlystopping = EarlyStopping(patience=self.configs.patience)
        model.train()
        for epoch in range(self.configs.epochs):
            out = model(x, edge_index)[mask_train]
            loss = criterion(out, label[mask_train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = self._calc_acc(out.argmax(-1), label[mask_train])
            val_loss, val_acc = self.valid(model, x, label, edge_index, mask_val, criterion)
            if epoch % 20 == 0:
                print(f'epoch: {epoch}, train loss: {loss.item()}, train acc: {acc*100}%\n '
                      f'\t\t vallid loss: {val_loss}, valid acc: {val_acc*100}%\n')
                earlystopping(val_loss, model, self.configs.model_path)

            if earlystopping.early_stop:
                print('Early stopping')
                # break

        test_loss, test_acc = self.test(model)
        print(f'test loss: {test_loss}, test acc: {test_acc*100}%\n')

    def test(self, model=None):
        data = self._get_data()
        x, label, edge_index, _, _, mask_test = data()
        x = x.to(self.device)
        label = label.to(self.device)
        edge_index = edge_index.to_dense().to(self.device).to_sparse()
        path = f"./checkpoints/{self.configs.model_path}"
        model = model or model.load_state_dict(torch.load(path)).to(self.device)
        criterion = self._select_criterion()
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)[mask_test]
            loss = criterion(out, label[mask_test])
        acc = self._calc_acc(out.argmax(-1), label[mask_test])
        print()
        return loss.cpu().numpy(), acc














