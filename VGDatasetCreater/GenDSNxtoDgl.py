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
    networkx 객체 만든거 가져와서 dgl 그래프로 변환 후
    데이터 셋에 graph list, label list, dim_nfeats(len(attr)), num_nodes, gclasses
    데이터셋 객체에 속성 부여
'''
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

from dgl.data import DGLDataset

with open("./data/networkx1000.pickle", "rb") as fr:
    networkXSet = pickle.load(fr)
    networkXSet = networkXSet[:1000]
with open("./data/clusterSifted1000.pickle", "rb") as fr:
    labels = pickle.load(fr)
#
# for nxGraph in networkXSet: #1000개 - label 개수 맞춰서
#     print(nxGraph.nodes[1][attr])
#     print(nxGraph)
#     dglGph = dgl.from_networkx(nxGraph, node_attrs=['attr', 'weight'])
#     print(dglGph.nodes[0].data['attr'])
#
#     break





class VGDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='VisualGenome')

    def process(self):
        # graph label, num_nodes 사용

        self.graphs = [] #networkX 객체 변환. 이거 dgl로 변환해서 append? 기존엔 str, dtn으로 생성했었음
        self.labels = labels
        self.dim_nfeats = 10

        #feature 개수.. 10개? self.feature

        # For each graph ID...


        for nxGraph in networkXSet: #1000개 - label 개수 맞춰서
            # Find the edges as well as the number of nodes and its label.
            #이미 src, dst는 id 값이 모두 relabel 되었음. src를 sort 하고 그에 맞춰서 edge를 바꿔야함. dict를 두 번?
            # 한 노드에 두 개의 노드가 연결된 경우가 있을 수 있음. 일반적인 dict 말고 tuple에서 iterable하게 만들어야 할 듯?
            #networkx to dgl 했을때 num_nodes, attr 유지되는 지 확인 후 nxto dgl 사용해서 graph 만들고 append, label은 그냥 바로 넣어도 될 듯?
            #어차피 순서대로니까

            dglGph = dgl.from_networkx(nxGraph, node_attrs=['attr'])
            #dglGph = dgl.from_networkx(nxGraph, node_attrs=['attr', 'weight'])


            self.graphs.append(dglGph)

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
