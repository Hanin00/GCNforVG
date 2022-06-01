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


with open("./data/networkx1000_noTensor.pickle", "rb") as fr:
    networkXSet = pickle.load(fr)
networkXSet = networkXSet[:1000]

with open("./data/clusterSifted1000.pickle", "rb") as fr:
    labels = pickle.load(fr)


class VGDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='VisualGenome')

    def process(self):
        # graph label, num_nodes 사용

        self.graphs = [] #networkX 객체 변환. 이거 dgl로 변환해서 append? 기존엔 str, dtn으로 생성했었음
        #self.labels = labels
        self.labels = labels
        self.dim_nfeats = 10

        for nxGraph in networkXSet: #1000개 - label 개수 맞춰서
            dglGph = dgl.from_networkx(nxGraph, node_attrs=['attr', 'weight'])
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
