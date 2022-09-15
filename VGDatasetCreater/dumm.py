import sys
import numpy as np
import pandas as pd
import torch
import csv

import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import FastText
from tqdm import tqdm
import time
import json
from collections import Counter
import pickle
import nltk
from nltk.corpus import conll2000









with open("./data/v3_x100/v3_x1000.pickle", "rb") as fr:
    graphs= pickle.load(fr)

print(len(graphs))
print(graphs[0])
print(graphs[1])
print(graphs[2])
print(graphs[3])

with open("./data/v3_x100/v3_x1001.pickle", "rb") as fr:
    graphs= pickle.load(fr)

print(len(graphs))
print(graphs[0])
print(graphs[1])
print(graphs[2])
print(graphs[3])



sys.exit()




def graphShowName(nexG):
    #nx.draw(nexG, with_labels=True)
    relabelDict = {}
    for idx in nexG.nodes():
        relabelDict[idx] = nexG.nodes[idx]['name']

    nexG = nx.relabel_nodes(nexG, relabelDict)
    plt.figure(figsize=[15, 7])
    nx.draw(nexG,  with_labels=True)
    plt.show()

def vGphShow(nexG):
    plt.figure(figsize=[15, 7])
    nx.draw(nexG, with_labels=True)
    plt.show()

def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

with open("data/v3_x100/v3_x1000.pickle", "rb") as fr:
    v3X1000 = pickle.load(fr)

with open("data/v3_x100/totalEmbDictV3_x100.pickle", "rb") as fr:
    embDict= pickle.load(fr)


# '''synset counter_top common value 1000'''
# with open("data/v3_x100/synsetDictV3_x100.pickle", "rb") as fr:
#     synDict= pickle.load(fr)
#
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
#
#
# synCnt = Counter(synDict.values())
# vals = [i[0] for i in synCnt.most_common()]
# valcnt = [i[1] for i in synCnt.most_common()]
# df = pd.DataFrame(vals, valcnt)
# print(df.head(1000))
#
#
# sys.exit()



nodeNameList = [

                [['road', 'traffic_light', 'car','vehicle'],
                 ['traffic_light', 'car','road','road']],

                [['traffic_light', 'road','car','car',],
                 ['fence', 'fence','road','fence', ]],

                [['headlight', 'road', 'road', 'sign' ],
                 ['car', 'car', 'boundary_line', 'road']],

                ]

gList = []
for i in range(len(nodeNameList)):
    objNodeName = nodeNameList[i][0]
    subNodeName = nodeNameList[i][1]

    df = pd.DataFrame({"objNodeName": objNodeName, "subNodeName": subNodeName, })
    gI = nx.from_pandas_edgelist(df, source='objNodeName', target='subNodeName')

    nodesList = objNodeName + subNodeName

    for index, row in df.iterrows():
        gI.nodes[row['objNodeName']]['name'] = row["objNodeName"]  # name attr
        gI.nodes[row['subNodeName']]['name'] = row['subNodeName']  # name attr

    for i in range(len(nodesList)):  # nodeId
        name = nodesList[i]
        emb = embDict[name]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
        for j in range(1):  # Embedding 값은 [3,]인데, 원소 각각을 특징으로 node에 할당
            nx.set_node_attributes(gI, {name: float(emb[j])}, "f" + str(j))

    dictIdx = {nodeId: idx for idx, nodeId in enumerate(nodesList)}
    gI = nx.relabel_nodes(gI, dictIdx)
    gList.append(gI)


with open("./data/query_road_0819.pickle", "wb") as fw:
    pickle.dump(gList, fw)

with open("./data/query_road_0819.pickle", "rb") as fr:
    gList = pickle.load(fr)


print(gList[0].nodes(data=True))
print(gList[1].nodes(data=True))
print(gList[2].nodes(data=True))



[graphShowName(graph) for graph in gList]



#graphShowName(gList[2])
