import sys
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
# import pandas as pd
import torch.optim as optim
import pickle
import torch.nn.functional as F
import numpy as np
from gensim.models import FastText
import torch.utils.data as utils
from torch.autograd import Variable

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

'''
    data.x : feature matrix[1000, num_node, 10]
    data.y : label 
    data.edge_list : sparse adjacency list -> local에 저장한 파일 사용
    data.batch : None 인 줄 알았는데, 그래프의 각 노드마다 해당 노드가 속한 그래프의 id를 넣어주면 되나봄
    
    Data(x=[19580, 3], edge_index=[2, 74564], y=[600])
    <class 'torch_geometric.data.data.Data'>

    해당 예제의 dataset type은 torch_geometric.data.data.Data임
    
    list를 tensor로 변경해서 넣어주면 될 듯
    
    x(Tensor) : node feature Matrix, shape : [num_nodes, num_node_features]
    edge_index(Long Tensor) : Graph connectivity format. shape : [2, num_edges]
    batch : None 인 줄 알았는데, 그래프의 각 노드마다 해당 노드가 속한 그래프의 id를 넣어주면 되나봄
    
    x : feature matrix (# of nodes, # of node feature dim) 
    edge_index : sparse adj list, 연결된 edge에 대한 node 저장  
          ex. node 1 : [1,4,6]
    batch : (array) batch마다 node 개수가 달라지므로 -> 어떤 node가 어떤 graph에 속하는지에 대한 정보 저장 
          ex. [1,1,1,1,1] : 5 nodes in graph 1 , [2,2,2] : 3 nodes in graph 2 

'''

# with open("./data/featMatrix1000_1.pickle", "rb") as fr:
#     featMall = pickle.load(fr)
# with open("./data/edgeList1000.pickle", "rb") as fr:
#     edgeList = pickle.load(fr)
# # with open("./data/edgeList1000.pickle", "rb") as fr:
# #     edge_index = pickle.load(fr)
#
# x = featMall
# # print(torch.tensor(x[1])) #그래프 하나에 대한 featrue matrix를 torch.tensor


edgeIdxAll = []
for i in range(len(edgeIdx)) :
    edgeIdxAll.append([torch.tensor(edgeIdx[i][0]),torch.tensor(edgeIdx[i][1])])

with open("./data/edgeListTensor.pickle", "wb") as fw:
    pickle.dump(edgeIdxAll, fw)

with open("./data/edgeListTensor.pickle", "rb") as fr:
    data = pickle.load(fr)

print(len(data))
print(type(data))
print(data.shape)

#edge_idx 이케 만들어야해~_~




# print(len(x[1]))
# print(len(x[1][0]))
# print(edgeIdx)


sys.exit()

#datset(Graph에 들어가야 하는거)
#그 다음에 GraphDataset 만들어서 써야하나봄..? 모람.. 이게..모야..




class GraphDataset(Dataset):
    def __init__(self, x_tensor, edge_index, batch, y_tensor, ):
        super(GraphDataset, self).__init__()
        self.x = x_tensor
        self.edge_index = edge_index
        self.batch = batch
        self.y = y_tensor.to(device)

    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    def num_node(self):
        data = self

    def __getitem__(self, index):
        # imgAdj = self.AdjList[idx]
        # label = self.classList[idx]
        imgAdj = self.x[index]
        label = self.y[index]
        attr = self.attr[index]
        return imgAdj, label, attr

    def __len__(self):
        return len(self.x)

#
#
# class GraphDataset(Dataset):
#     def __init__(self, x_tensor,edge_index, batch, y_tensor, ):
#         super(GraphDataset, self).__init__()
#         self.x = x_tensor
#         self.edge_index = edge_index
#         self.batch = batch
#         self.y = y_tensor.to(device)
#
#
#
#         self.name = name
#         self.cleaned = cleaned
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#         if self.data.x is not None and not use_node_attr:
#             num_node_attributes = self.num_node_attributes
#             self.data.x = self.data.x[:, num_node_attributes:]
#         if self.data.edge_attr is not None and not use_edge_attr:
#             num_edge_attributes = self.num_edge_attributes
#             self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
#
#
#
#         y_one_hot = torch.zeros((len(y_tensor), 15)).to(device)
#         yUnsqueeze = y_tensor.unsqueeze(1).to(device)
#         print(yUnsqueeze.is_cuda)
#
#         y_one_hot.scatter_(1, yUnsqueeze, 1)
#
#         self.attr = y_one_hot
#         self.AdjList = []
#         self.classList = []
#
#     def num_node_features(self) -> int:
#         r"""Returns the number of features per node in the dataset."""
#         data = self[0]
#         data = data[0] if isinstance(data, tuple) else data
#         if hasattr(data, 'num_node_features'):
#             return data.num_node_features
#         raise AttributeError(f"'{data.__class__.__name__}' object has no "
#                              f"attribute 'num_node_features'")
#
#     def num_node(self) :
#         data = self[]
#
#
#     def __getitem__(self, index):
#         # imgAdj = self.AdjList[idx]
#         # label = self.classList[idx]
#         imgAdj = self.x[index]
#         label = self.y[index]
#         attr = self.attr[index]
#         return imgAdj, label, attr
#
#     def __len__(self):
#         return len(self.x)
#
#
# if __name__ == "__main__":
#     print(len(Images))
#     print(len(labels))
#     dataset = GraphDataset(Images, labels)
#     dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)
#
#     for epoch in range(2):
#         print(f"epoch : {epoch}")
#         for adj, label, attr in dataloader:
#             print("epoch", epoch, "label : ", label, "onehot ,attr : ", attr)
