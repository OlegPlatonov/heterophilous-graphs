import os
import numpy as np
import torch
from torch.nn import functional as F
import dgl
from sklearn.metrics import roc_auc_score


class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu'):
        print('Preparing data...')
        data = np.load(os.path.join('data', name.replace('-', '_')))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)
        graph = dgl.to_bidirected(graph)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        train_idx_list = [torch.where(train_mask)[0] for train_mask in data['train_masks']]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in data['val_masks']]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in data['test_mask']]

        # add node feature augmentation here

        self.name = name
        self.device = device

        self.graph = graph.to(device)
        self.node_features = node_features.to(device)
        self.labels = labels.to(device)

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits

    def compute_metrics(self, logits):
        if self.num_targets == 1:
            train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
                                         y_score=logits[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
                                       y_score=logits[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
                                        y_score=logits[self.test_idx].cpu().numpy()).item()

        else:
            preds = logits.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics
