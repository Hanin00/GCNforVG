import torch
import networkx as nx
import sys
import time
#from torch_geometric.datasets import TUDataset
#import torch_geometric.utils as pyg_utils
from tqdm import tqdm
import matplotlib.pyplot as plt

# a = -3
# b = -4
# dataset = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
# dataset = list(dataset)
# print(dataset[a])
# print(dataset[b])
# print(dataset[a].edge_index)
# data = []
# for i, graph in tqdm(enumerate(dataset)):
#     if not type(graph) == nx.Graph:
#         graph = pyg_utils.to_networkx(graph).to_undirected()
#         data.append(graph)
# print(data[a])
# print(data[b])
# print("-----------------------------")
# start = time.time()
# G = nx.optimize_graph_edit_distance(data[a], data[b])
# for i in G:
#     print(i)
# print("time : ", time.time()-start)
'''
# 첫번째 실험 [0개, 0.069초]
Data(edge_index=[2, 18], x=[5, 3], y=[1])
Data(edge_index=[2, 18], x=[5, 3], y=[1])
# 두번째 실험 [7개, 71140.000초//약 49시간]
Data(edge_index=[2, 58], x=[16, 3], y=[1])
Data(edge_index=[2, 54], x=[15, 3], y=[1])
'''

'''
nx.graph_edit_distance시
node의 label을 고려함
node의 attribute를 고려하지 않음
edge의 attribute를 고려함


#https://frhyme.github.io/python-lib/nx_graph_relabeling/
'''
G1 = nx.cycle_graph(4)
G2 = nx.cycle_graph(6)

# G1.add_nodes_from([(2, {"color": "white"}), (3, {"color": "green"})])
# G2.add_nodes_from([(0, {"color": "blue"}), (3, {"color": "green"})])

#G2.remove_edge(0,1)

#mapping = {0: "a", 1: "b", 2: "c"}
mapping = {0 : 7, 1:8, 2: 9, 3:10, 4:11, 5:12}
G3 = nx.relabel_nodes(G2, mapping)
# nx.draw(G3,with_labels=True)
# plt.savefig("./data/G3.png")
G2.add_edges_from([(0, 3, {"color": "white"}), (1, 3, {"color": "green"})])
G3.add_edges_from([(7, 8, {"color": "white"}), (8, 9, {"color": "green"})])
print(nx.graph_edit_distance(G1, G2))
print(nx.graph_edit_distance(G2, G3))

#print(nx.nodes(G2))
#print(nx.nodes(G3))
print(nx.edges(G2))
#print(nx.edges(G3))
print(G3.edges())
print(type(G3.edges()))


#print(nx.graph_edit_distance(G2,G3,edge_match=(G2[0][1], G3[2][2]),edge_subst_cost=1))

#nx.draw(G1,with_labels=True)
#nx.draw(G2,with_labels=True)

#plt.savefig("./data/G3.png")




# G2.add_node(10)
# G2.add_edge(1, 10)

# G1.add_edges_from([(2, 3, {"color": "white"}), (1, 3, {"color": "green"})])
# G2.add_edges_from([(0, 3, {"color": "white"}), (1, 3, {"color": "green"})])
# print(G1.edges.data())
# print(G2.edges.data())

