import json
import pickle


# scene_graph.json -> adjMatrix/FeatureMatrix 만들 때 오류나서 제외한 사진들이 있음
# 해당 이미지를 제외하고 label 값을 추출해야해서 따로 처리함
def siftCluster():
    testFile = open('./data/cluster5000.txt', 'r')
    readFile = testFile.readline()
    oldLabel = (readFile[1:-1].replace("'", '')).split(',')
    label = []

    siftListId = [50, 60, 156, 241, 284, 299, 317, 371, 403, 432, 512, 520, 647, 677, 745, 867, 930, 931, 1102, 1116,
                  1136, 1174, 1212, 1239, 1256, 1278]

    for i in range(len(oldLabel)):
        if i in siftListId:
            continue
        else:
            label.append(i)
        if (len(label) == 1000):
            break

    print(len(label))
    with open("./data/clusterSifted1000.pickle", "wb") as fw:
        pickle.dump(label, fw)


# siftCluster()


# netx에서 adjMatrix sparse 형태로 뽑아서 edge list로 사용하기
def tupleToList():
    return


''''
    data
    data.y : cluster 값, list 형태로
    data.y[mask] : cluster label 이라는데 일단 data.y[mask] = data.y?
    data.val_mask : random seed 정해서 비율 idx 범위정하기?
    data.train_mask :
    data.test_mask : 
    
    
    data.x : feature matrix
    data.edge_list : sparse adjacency list
    data.batch : batch vector... ????? 그래프 노드 개수??? 그래프??? 그래프들? vector라는데 networkx 객체가 아닌건가?? Long tensor라는데 이게 뭐지.. < optional 이라는데
    
    data.num_nodes
    data.num_classes : 클래스 개수, 10
    data.num_graphs : 그래프 개수, 1000 

    data.num_node_features  << 없을 수 있음    
    
    data.test_neg_edge_index / test_pos_edge_index    
    data.train_neg_edge_index / train_pos_edge_index    
'''
import time
import networkx as nx
import numpy as np
import sys
import pandas as pd

with open("./data/networkx1000.pickle", "rb") as fr:
    netx = pickle.load(fr)

def mkEdgelistPer1(netx):
    a = netx.edges.data()
    sr = pd.DataFrame(a)
    sr.columns = ['n1', 'n2', 'feature']
    n1 = sr['n1'].values.tolist()
    n2 = sr['n2'].values.tolist()
    return n1, n2

n1, n2 = mkEdgelistPer1(netx[0])  #local 저장?
print(n1, n2)


def mkEdgelistPer1(netx):
    a = netx.edges.data()
    sr = pd.DataFrame(a)
    sr.columns = ['n1', 'n2', 'feature']
    n1 = sr['n1'].values.tolist()
    n2 = sr['n2'].values.tolist()
    return n1, n2
