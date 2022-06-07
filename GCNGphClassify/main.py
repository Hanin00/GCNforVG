import networkx as nx
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from VGDatasetCreater import GenDSNxtoDgl as gds
import model as md
import random
from tqdm import tqdm
import numpy as np
import sys

with open("data/networkx1000_attr.pickle", "rb") as fr:
    netX = pickle.load(fr)
netX = netX[:1000]
with open("./data/clusterSifted1000.pickle", "rb") as fr:
    labels = pickle.load(fr)

adjMs = []
for graph in netX:
    adjMs.append(nx.adjacency_matrix(graph).todense())
#    featMs.append(graph.attr_matrix.todense())

labels = torch.LongTensor(labels)

#print(adjMs[0])
#print(adjMs[0].todense()) #<class 'numpy.matrix'>

# gpu 사용
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# todo input-output / flatten(weight) / 일단 10개만 학습해보기
dataset = gds
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

num_examples = len(dataset)
num_train = int(num_examples * 0.7)

train_sampler = SubsetRandomSampler(torch.arange(num_train).to(device))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples).to(device))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=1, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False)

# it = iter(train_dataloader)
# batch = next(it)

model = md.GCN  # n_features = 100, n_labels = 15
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-10)
# labels, features, model = labels.to(device), features.to(device), model.to(device)


for epoch in range(20):
    '''
    1-1. to dense 해서 일일히 넣어줌
    1-2 . GCN 모델에서 flatten
    
    2. 전체 셋을 넣고, GCN 모델에서 하나씩 까서 넣어줌. 
        -> 이게 모델 취지에 더 맞는 형태 같긴 한데..
    '''
    # batched_graph : 1,100,100, labels :    attr : 1,15
    for batched_graph, labels, attr in train_dataloader:
        model.train()
        batched_graph, labels, attr = batched_graph.to(device), labels.to(device), attr.to(device)
        batched_graph = batched_graph.squeeze().to(device)

        n_labels = 15  # 10
        n_features = features.shape[1]  # 10  #features = Tensor(100,10)
        pred = model(n_features, n_labels, batched_graph).to(device)  # Tensor(1,100,100), features = Tensor(100,10)

        # loss = F.nll_loss(pred[0], attr.squeeze().long()).to(device)
        loss = F.nll_loss(pred[0], attr.squeeze().long()).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0

for batched_graph, labels, attr in test_dataloader:
    model.eval()
    batched_graph, labels, attr = batched_graph.to(device), labels.to(device), attr.to(device)
    batched_graph = batched_graph.squeeze().to(device)
    attr = attr.squeeze().long().to(device)
    pred = model(n_features, n_labels, batched_graph).to(device)

    num_correct += (pred == attr.squeeze().long()).sum().item()

    # print("Test : pred.max : ", pred[0], "labels : ", labels, "attr : ", attr)
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)