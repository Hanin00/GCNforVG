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


# conllDict = {word: tag for word, tag in conll2000.tagged_words(tagset='universal')}
#
# print(conllDict['power'])
# print(conllDict['line'])
# print(conllDict['power_line'])
#
#
#
#

# with open("data/networkx_ver2.pickle", "rb") as fr:
#     ver2G = pickle.load(fr)
#
#
# for i in range(len(ver2G)) :
#     names = [row[1] for row in ver2G[i].nodes(data='name')]
#     if 'power_line' in names :
#         print(i)
#         print(ver2G[i].nodes(data=True))
#         break




with open("data/networkx_ver200.pickle", "rb") as fr:
    n200G = pickle.load(fr)


# for i in range(len(n200G)) :
#     names = [row[1] for row in n200G[i].nodes(data='name')]
#     if 'grass' in names :
#         print(n200G[i].nodes(data=True))
#         break


def vGphShow(nexG):
    # nx.draw(nexG, with_labels=True)

    plt.figure(figsize=[15, 7])
    nx.draw(nexG, with_labels=True)
    plt.show()


with open("data/networkx_ver1.pickle", "rb") as fr:
    net200 = pickle.load(fr)

# for i in range(len(net200)) :
#     names = [row[1] for row in net200[i].nodes(data='name')]
#     if 'truck' in names :
#         print(net200[i].nodes(data=True))
#         break
#
# nodeNames = ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', 'road',
#              'window', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'truck', 'leaf', 'tree',
#              'chair', 'rock', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'person', 'tree', 'chair',
#              'person', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'rock', 'leaf', 'lamp', 'keyboard',
#              'monitor',
#              'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair',
#              'keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair', 'wall',
#              'window', 'table', 'light', 'book', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling',
#              'table', 'light', 'book', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'paper',
#              'man', 'shirt', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'wall', 'person'
#     , 'shirt', 'bag', 'water', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'jacket']
#
# nodeNames = list(set(nodeNames))
#
# embDict = {}
# for i in range(len(net200)):
#     names = [row[1] for row in net200[i].nodes(data='name')]
#     f0 = [row[1] for row in net200[i].nodes(data='f0')]
#     f1 = [row[1] for row in net200[i].nodes(data='f1')]
#     f2 = [row[1] for row in net200[i].nodes(data='f2')]
#
#     for i in range(len(names)):
#         features = [f0[i], f1[i], f2[i]]
#         embDict[names[i]] = features
#
# nodeNameList = [
#     [['truck', 'building', 'car', 'tree', 'man', 'building', 'car', 'tree', 'man', ],
#      ['truck', 'building', 'car', 'tree', 'man', 'truck', 'building', 'car', 'tree']],
#
#     [['leaf', 'tree', 'flower', 'cloud', 'grass', 'bench', 'cloud', 'grass', 'bench'],
#      ['flower', 'cloud', 'grass', 'bench', 'leaf', 'tree', 'flower', 'flower', 'cloud', ]],
#
#     [['sky', 'person', 'land', 'bench', 'sky', 'person', 'land', 'bench', 'sky', 'person'],
#      ['person', 'land', 'bench', 'sky', 'person', 'bench', 'sky', 'person', 'bench', 'sky', ]],
#
#     [['lamp', 'keyboard', 'monitor', 'cabinet', 'cup', 'lamp', 'keyboard', 'monitor', 'cabinet', 'cup', ],
#      ['cabinet', 'cup', 'keyboard', 'monitor', 'cabinet', 'cup', 'lamp', 'keyboard', 'monitor', 'cabinet', ]],
#
#     [['rug', 'curtain', 'desk', 'window', 'book', 'curtain', 'desk', 'window', 'book'],
#      ['desk', 'window', 'book', 'rug', 'curtain', 'rug', 'curtain', 'desk', 'window', ]],
#
#     [['chair', 'desk', 'window', 'book', 'curtain', 'desk', 'window', 'book', 'curtain', ],
#      ['book', 'curtain', 'chair', 'desk', 'window', 'chair', 'desk', 'window', 'book', ]],
#
#     [['keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'cabinet', 'cup', 'rug', 'keyboard', 'monitor', ],
#      ['rug', 'keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'keyboard', 'monitor', 'cabinet', 'cup', ]],
#
#     [['curtain', 'desk', 'window', 'book', 'chair', 'curtain', 'desk', 'window', 'book', 'chair', ],
#      ['window', 'book', 'chair', 'curtain', 'desk', 'book', 'chair', 'curtain', 'desk', 'window', ]],
#
#     [['wall', 'keyboard', 'monitor', 'cabinet', 'cup', 'wall', 'keyboard', 'monitor', 'cabinet', 'cup', ],
#      ['cabinet', 'cup', 'wall', 'keyboard', 'monitor', 'cabinet', 'cup', 'wall', 'keyboard', 'monitor', ]],
#
#     [['window', 'table', 'light', 'book', 'desk', 'pillow', 'window', 'table', 'light', 'book', 'desk',
#       'pillow', ],
#      ['book', 'desk', 'pillow', 'window', 'table', 'light', 'desk', 'pillow', 'window', 'window', 'table',
#       'table']],
#
#     [['letter', 'book', 'cup', 'bottle', 'ceiling', 'letter', 'book', 'cup', 'bottle', 'ceiling', ],
#      ['bottle', 'ceiling', 'letter', 'book', 'cup', 'book', 'cup', 'ceiling', 'ceiling', 'bottle', ]],
#
#     [['table', 'light', 'book', 'desk', 'pillow', 'letter', 'table', 'light', 'book', 'desk', 'pillow',
#       'letter', ],
#      ['book', 'book', 'pillow', 'pillow', 'desk', 'desk', 'pillow', 'letter', 'letter', 'letter', 'table',
#       'light', ]],
#
#     [['man', 'shirt' ,'bag', 'water', 'hat', 'mirror', 'ceiling', 'man', 'shirt', 'bag', 'water', 'hat', ],
#      ['water', 'water', 'water', 'mirror', 'mirror', 'ceiling', 'hat', 'mirror', 'ceiling', 'ceiling', 'man', 'shirt']],
# ]
#
# gList = []
# for i in range(len(nodeNameList)):
#     objNodeName = nodeNameList[i][0]
#     subNodeName = nodeNameList[i][1]
#
#     df = pd.DataFrame({"objNodeName": objNodeName, "subNodeName": subNodeName, })
#     gI = nx.from_pandas_edgelist(df, source='objNodeName', target='subNodeName')
#
#     nodesList = objNodeName + subNodeName
#
#     for index, row in df.iterrows():
#         gI.nodes[row['objNodeName']]['name'] = row["objNodeName"]  # name attr
#         gI.nodes[row['subNodeName']]['name'] = row['subNodeName']  # name attr
#
#     for i in range(len(nodesList)):  # nodeId
#         name = nodesList[i]
#         emb = embDict[name]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
#         for j in range(3):  # Embedding 값은 [3,]인데, 원소 각각을 특징으로 node에 할당
#             nx.set_node_attributes(gI, {name: float(emb[j])}, "f" + str(j))
#
#     dictIdx = {nodeId: idx for idx, nodeId in enumerate(nodesList)}
#     gI = nx.relabel_nodes(gI, dictIdx)
#     gList.append(gI)
#

# with open("./data/test12.pickle", "wb") as fw:
#     pickle.dump(gList, fw)
#
# with open("data/test12.pickle", "rb") as fr:
#     gList = pickle.load(fr)
#
#
#
# for idx in range(len(gList)) :
#     i = gList[idx]
#     nodeIdList = []
#     nameList = []
#     lista = i.nodes(data='name')
#
#     for j in lista :
#         nodeIdList.append(j[0])
#         nameList.append(j[1])
#
#     print('graph Id : ', idx )
#     df = pd.DataFrame({"id":nodeIdList, "name" : nameList})
#     print(df)


#print(gList[0].nodes(data=True))

