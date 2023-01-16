import torch
import numpy as np
import torch.nn.functional as F
from models import GCN, GCL
from graph_learners import FGP_learner, ATT_learner, MLP_learner, GNN_learner
from utils import cal_accuracy, get_masked_features, normalize, split_batch
from GSL.data_loader import Cora


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def load_data(self):
        # TODO: Other datasets
        data_dict = {
            "Cora": Cora
        }
        dataset = data_dict[self.configs.dataset](self.configs.root_path)
        features, in_features, label, adj, masks, n_classes = dataset()
        return features, in_features, label, adj, masks, n_classes

    def cal_cls_loss(self, model, mask, adj, features, labels):
        out = model(features, adj)
        loss = F.cross_entropy(out[mask], labels[mask])
        acc = cal_accuracy(out, features)
        return loss, acc

    def cal_loss_gcl(self, model, graph_learner, features, anchor_adj):
        """anchor view"""
        aug_feats_anchor = get_masked_features(features, self.configs.maskfeat_prob_anchor)
        z_anchor, _ = model(aug_feats_anchor, anchor_adj)

        """learner view"""
        learner_adj = graph_learner(features)
        learner_adj = (learner_adj + learner_adj.t()) / 2.  # symmetrize
        learner_adj = normalize(learner_adj)

        aug_feats_learner = get_masked_features(features, self.configs.maskfeat_prob_learner)
        z_learner, _ = model(aug_feats_learner, learner_adj)

        if self.configs.contract_batch_size > 0:
            nodes_idx = list(range(features.shape[0]))
            batches = split_batch(nodes_idx, self.configs.contract_batch_size)
            loss = 0
            for batch in batches:
                loss += model.cal_gcl_loss(z_anchor[batch], z_learner[batch]) * len(batch) / features.shape[0]
        else:
            loss = model.cal_gcl_loss(z_anchor, z_learner, temperature=self.configs.temperature)
        return loss, learner_adj

    def evaluate_adj_by_cls(self, adj, features, in_features, labels, n_classes, masks):
        """masks = (train, val, test)"""
        device = features.device
        cls_model = GCN(in_features, self.configs.hidden_features_cls, n_classes, self.configs.n_layers_cls,
                        self.configs.dropout_node_cls, self.configs.dropout_edge_cls).to(device)
        optimizer = torch.optim.Adam(cls_model.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls)

        best_acc = 0.
        early_stop_count = 0
        best_model = None

        print("--------------------------Evaluate_Adj_By_Cls-------------------------")
        print("--------------------------Training-------------------------")
        cls_model.train()
        for epoch in range(self.configs.epochs_cls):
            loss, acc = self.cal_cls_loss(cls_model, masks[0], adj, features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}")
            if epoch % 10 == 0:
                cls_model.eval()
                val_loss, acc = self.cal_cls_loss(cls_model, masks[1], adj, features, labels)
                print(f"Epoch {epoch}: val_loss={val_loss.item()}, val_accuracy={acc}")
                if acc > best_acc:
                    early_stop_count = 0
                    best_acc = acc
                    # torch.save(cls_model, self.configs.save_path_cls)
                    best_model = cls_model
                else:
                    early_stop_count += 1
                    print(f"Early Stopping Count={early_stop_count}/{self.configs.patience_cls}")
                if early_stop_count >= self.configs.patience_cls:
                    print("Early Stopping")
                    break
        print("--------------------------Testing-------------------------")
        best_model.eval()
        test_loss, test_acc = self.cal_cls_loss(best_model, masks[2], adj, features, labels)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        return best_acc, test_acc, best_model

    def train(self):
        device = self.device
        if self.configs.gsl_mode == 'structure_inference':
            features, in_features, label, adj, masks, n_classes = self.load_data()

        # TODO structure_refinement datasets
        elif self.configs.gsl_mode == 'structure_refinement':
            pass

        if self.configs.downstream_task == 'classification':
            best_val = 0.
            best_val_test = 0.

        # TODO clustering task
        elif self.configs.downstream_task == 'clustering':
            pass

        if self.configs.learner_type == 'fgp':
            graph_learner = FGP_learner(features, self.configs.k_neighbours, self.configs.knn_metric,
                                        self.configs.slope, self.configs.alpha)
        elif self.configs.learner_type == 'mlp':
            graph_learner = MLP_learner(self.configs.n_layers_gsl, in_features, in_features, self.configs.k_neighbours, self.configs.knn_metric,
                                        self.configs.slope, self.configs.alpha)
        elif self.configs.learner_type == 'att':
            graph_learner = ATT_learner(self.configs.n_layers_gsl, in_features, self.configs.k_neighbours, self.configs.knn_metric,
                                        self.configs.slope, self.configs.alpha)
        elif self.configs.learner_type == 'gnn':
            graph_learner = GNN_learner(self.configs.n_layers_gsl, in_features, self.configs.k_neighbours, self.configs.knn_metric,
                                        self.configs.slope, self.configs.alpha)
        graph_learner = graph_learner.to(device)
        model = GCL(self.configs.n_layers_gcl, in_features, self.configs.hidden_features, self.configs.embed_features,
                    self.configs.proj_features, self.configs.dropout_node, self.configs.dropout_edge).to(device)

        optimizer_gcl = torch.optim.Adam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
        optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)

        features = features.to(device)
        label = label.to(device)
        # anchor_adj = torch.sparse_coo_tensor(torch.arange(features.shape[0]).repeat(2, 1),
        #                                      torch.ones(features.shape[0]), size=adj.shape)
        anchor_adj = torch.eye(features.shape[0])
        anchor_adj = normalize(anchor_adj, mode='sym').to(device)

        print("--------------------------Train SUBLIME-------------------------")
        model.train()
        graph_learner.train()
        for epoch in range(self.configs.epochs):
            loss, adj = self.cal_loss_gcl(model, graph_learner, features, anchor_adj)
            optimizer_gcl.zero_grad()
            optimizer_learner.zero_grad()
            loss.backward()
            optimizer_gcl.step()
            optimizer_learner.step()

            if (1 - self.configs.tau) and (self.configs.iterations == 0 or epoch % self.configs.iterations == 0):
                anchor_adj = self.configs.tau * anchor_adj + (1 - self.configs.tau) * adj.detach()

            print(f"Epoch {epoch}: train_loss={loss.item()}")

            if epoch % self.configs.eval_freq == 0:
                if self.configs.downstream_task == 'classification':
                    model.eval()
                    graph_learner.eval()
                    val_acc, test_acc, _ = self.evaluate_adj_by_cls(adj, features, in_features, label, n_classes, masks)
                    print(f"Epoch {epoch}: val_accuracy={val_acc.item()}, test_accuracy={test_acc}")
                    if val_acc > best_val:
                        best_val = val_acc
                        best_val_test = test_acc

                # TODO: clustering task
                elif self.configs.downstream_task == 'clustering':
                    pass
        if self.configs.downstream_task == 'classification':
            print(f"best_val_accuracy={best_val.item()}, best_test_accuracy={best_val_test}")




        













