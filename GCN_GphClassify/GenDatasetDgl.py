import sys
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.optim as optim
import pickle
import torch.nn.functional as F
import numpy as np
from gensim.models import FastText
import torch.utils.data as utils
from torch.autograd import Variable
import pandas as pd
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl
import tqdm



'''
dgl에서는 convert.py를 건드려서 num_node를 실제 값(csv에 있는)과 max_id 를 비교하는 문제를 해결할 수 있었음
V dataFrame(<heterograph.py)도 고쳐야한다는 문제 발생
-> 이때 실제 원본 데이터 보면 각 그래프마다 노드가 0부터 시작한다는 걸 발견
-> networkx 폴더에서 networkx 객체의 node를 relabel 하는 방법으로 변경

'''

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

edges = pd.read_csv('./data/graph_edges.csv', encoding='cp949')
properties = pd.read_csv("./data/graph_properties.csv", encoding='cp949')

from dgl.data import DGLDataset
# class VGDataset(DGLDataset):
class VGDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='VisualGenome')

    def process(self):
        edges = pd.read_csv('./data/graph_edges.csv')
        properties = pd.read_csv("./data/graph_properties.csv")
        with open("./data/featMatrix1000_1.pickle", "rb") as fr:
            features = pickle.load(fr)
        self.graphs = []
        self.labels = []
        self.dim_nfeats = 10

        #feature 개수.. 10개? self.feature

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]


            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)
        self.gclasses = int(max(self.labels)+1)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


# dataset = VGDataset()
# graph, label = dataset[0]
#print(graph, label)

# with open("./data/VGDataset.pickle", "wb") as fw:
#     pickle.dump(dataset, fw)
#
# with open("./data/VGDataset.pickle", "rb") as fr:
#     data = pickle.load(fr)
