import torch
import torch.nn as nn
import torch.nn.functional as F

# Graph convolution model
import torch_geometric.nn as pyg_nn
# Graph utility function
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

# 사용할 데이터셋
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
data = dataset.data
batch = dataset.data.batch

# 이미 layer가 구현되어 있으므로, stacking만 하면 된다.
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        # ModuleList(): 각 레이어를 리스트에 전달하고 레이어의 iterator를 만든다.
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))

        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':  # node classification
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:  # graph convolution
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        """
        x: feature matrix \in R ^(# of nodes \times d(embedding dimension))
        edge_index : sparse adjacency list
        ex) node1: [1,4,6]
        batch: batch마다 node의 개수가 다름. => 매우 복잡

        """

        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:  # no features # feature가 없는 경우
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  # Modulelist # convolution layer
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout,
                          training=self.training)  # testing에는 사용 x # nn.Dropout()을 사용하면 이럴 필요 없을텐데,, 굳이,,?
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        # graph classification인 경우, pooling이 필요하므로 !!
        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)  # max

        x = self.post_mp(x)  # Sequential MLP

        return emb, F.log_softmax(x,
                                  dim=1)  # emd: to visualize that graph looks like # F.lof_softmax: CROSS-ENTROPY를 위해

    def loss(self, pred, label):
        return F.nll_loss(pred, label)  # nn.CrossEntropyLoss는 nn.LogSoftmax()와 nn.NLLLoss() # label: one-hot


class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')  # "Add" aggregation. # mean, max 등등
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)  # convolution

    def forward(self, x, edge_index):
        """
        Convolution을 위해서는 2가지가 필수적임.
        x has shape [N, in_channels] # feature matrix
        edge_index has shape [2, E] ==> connectivity ==> 2: (u, v)

        """

        # Add self-loops to the adjacency matrix.
        # A -> \tilde{A}
        # pyg_utils.add_self_loops(edge_index, num_nodes = x.size(0))
        # neighbor 정보뿐만 아니라, 내 정보까지 add해야하므로 self-loops 추가!

        # 지울수도 있다 !
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Transform node feature matrix.
        self_x = self.lin_self(x)  # B
        x = self.lin(x)  # W

        # self_x: skip connection #compute message for all the nodes
        return self_x + self.propagate(edge_index,
                                       size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # Compute messages
        # x_i is self-embedding
        # x_j has shape [E, out_channels]

        row, col = edge_index
        deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        F.normalize(aggr_out, p=2, dim=-1)  # dim: 상황에 따라 알맞게 조정할
        return aggr_out

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':  # node classification
            return CustomConv(input_dim, hidden_dim)


def train(dataset, task, writer):
    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # build model
    model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
    opt = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(200):
        total_loss = 0
        model.train()
        for batch in loader:
            # print(batch.train_mask, '----')
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs  ##
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)  # tensorboard

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)  # tensorboard

    return model


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total

writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset = dataset.shuffle()
task = 'graph'

model = train(dataset, task, writer)

color_list = ["red", "orange", "green", "blue", "purple", "brown"]

loader = DataLoader(dataset, batch_size=64, shuffle=True)
embs = []
colors = []
for batch in loader:
    emb, pred = model(batch)
    embs.append(emb)
    colors += [color_list[y-1] for y in batch.y] # True label에 해당하는 color 선
embs = torch.cat(embs, dim=0)

xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
plt.scatter(xs, ys, color=colors)
