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

# with open("data/networkx_ver2.pickle", "rb") as fr:
#     net200 = pickle.load(fr)


'''
    query graph 생성 - 많이 사용된 name을 기준으로 10개에서 15개의 노드를 갖는 그래프 5개
    특징값 : synset dict와 textEmbedding 이용해서 만들면 됨    
'''

# with open("data/networkx_ver200.pickle", "rb") as fr:
#     n200G = pickle.load(fr)
#
#
for i in range(len(n200G)) :
    names = [row[1] for row in n200G[i].nodes(data='name')]
    if 'grass' in names :
        print(n200G[i].nodes(data=True))
        break



# with open("data/synsetDict_1000.pickle", "rb") as fr:
#     synsDict = pickle.load(fr)
#
# n1000 = synsDict.values()
# cnt1000 = Counter(n1000)
# tu = cnt1000.most_common(100)
#
# pd.set_option('display.max_rows',None)
# df = pd.DataFrame(tu, columns=['Name', 'cnt'])
# print(df)

def vGphShow(nexG):
    #nx.draw(nexG, with_labels=True)

    plt.figure(figsize=[15, 7])
    nx.draw(nexG,  with_labels=True)
    plt.show()

with open("data/networkx_ver1.pickle", "rb") as fr:
    net200 = pickle.load(fr)
#
# for i in range(len(net200)) :
#     names = [row[1] for row in net200[i].nodes(data='name')]
#     if 'truck' in names :
#         print(net200[i].nodes(data=True))
#         break


# ---------- test data 생성. synset Dict. value 기준으로 다수 언급되는 100개의 단어 중에서 일반적인 사진을 묘사하도록 노드 선정 및 배치 vvv
# 기존 ver 1의 특징 값을 갖도록 설정함. 각 그래프의 노드 개수는 10~15개 사이
# nodeNames = ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', 'road',
#              'window', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'truck', 'leaf', 'tree',
#              'chair', 'rock', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky','person', 'tree', 'chair',
#              'person', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'rock', 'leaf','lamp', 'keyboard', 'monitor',
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
#     for i in range(len(names)) :
#         features = [f0[i],f1[i], f2[i]]
#         embDict[names[i]] = features
#
#
#
# nodeNameList = [[['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', 'street', 'road', 'bicycle',],
#                  ['road', 'window', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'truck','building', 'car', 'tree',]],
#
#                 [['leaf', 'tree', 'chair', 'rock', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'person','land', 'bench', 'sky', 'person' ],
#                  ['tree', 'chair', 'person', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'rock', 'leaf','tree', 'chair', 'person', 'flower',]],
#
#                 [['lamp', 'keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book',
#                   'chair', 'desk', 'window', 'book', 'chair',],
#                  ['keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair',
#                   'wall','keyboard', 'monitor', 'cabinet', 'cup',  ]],
#
#                 [['window', 'table', 'light', 'book', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'book', 'cup', 'bottle', 'ceiling'],
#                  ['table', 'light', 'book', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'paper','table', 'light', 'book', 'desk',]],
#
#                 [['man', 'shirt', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'wall', 'person', 'ceiling', 'leg', 'wall', 'person'],
#                  ['shirt', 'bag', 'water', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'jacket', 'mirror', 'seat', 'ceiling', 'leg']],
#                 ]
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
#
# print(gList[0])
# print(gList[0].nodes(data=True))
# vGphShow(gList[0])
# vGphShow(gList[1])
# vGphShow(gList[2])
# vGphShow(gList[3])
# vGphShow(gList[4])
#
# with open("./data/test5.pickle", "wb") as fw:
#     pickle.dump(gList, fw)
## ---------- test data 생성. synset Dict. value 기준으로 다수 언급되는 100개의 단어 중에서 일반적인 사진을 묘사하도록 노드 선정 및 배치^^^
# feature name dict


with open("data/test5.pickle", "rb") as fr:
    graphs = pickle.load(fr)





# with open("data/synsetDict_1000.pickle", "rb") as fr:
#     synsDict = pickle.load(fr)
#


# with open("data/totalEmbDict.pickle", "rb") as fr:
#     tModel = pickle.load(fr)
#
# print(tModel.get_sentence_vector['man'])


'''
    1. 1000개, 200개 동일한 특징 노드에 클래스의 특징이 같은 지 확인
    2. 1000개 senset과 1200개 까지의 synset에서 다른 게 몇 개 인지?
'''
# with open("data/networkx_ver2.pickle", "rb") as fr:
#     netv2G = pickle.load(fr)
#
# with open("data/networkx_ver200.pickle", "rb") as fr:
#     n200G = pickle.load(fr)
#
#
# for i in range(len(netv2G)) :
#     names = [row[1] for row in netv2G[i].nodes(data='name')]
#     if 'grass' in names :
#         g1Id = i
#         break
#
# for i in range(len(n200G)) :
#     names = [row[1] for row in n200G[i].nodes(data='name')]
#     if 'grass' in names :
#         g2Id = i
#         break
#
#
# G1 = netv2G[g1Id]
# G2 = n200G[g2Id]
#
# f0List1 = [row[1] for row in G1.nodes(data='f0')]
# f1List1 = [row[1] for row in G1.nodes(data='f1')]
# f2List1 = [row[1] for row in G1.nodes(data='f2')]
# name1 = [row[1] for row in G1.nodes(data='name')]
#
# f0List2 = [row[1] for row in G2.nodes(data='f0')]
# f1List2 = [row[1] for row in G2.nodes(data='f1')]
# f2List2 = [row[1] for row in G2.nodes(data='f2')]
# name2 = [row[1] for row in G2.nodes(data='name')]
#
#
# pd.set_option('display.max_rows', 10000)
# pd.set_option('display.max_columns', 10000)
#
# df = pd.DataFrame({ 'f0List1' : f0List1[:10],'f1List1' : f1List1[:10], 'f2List1' : f2List1[:10],
#                     'f0List2' : f0List2[:10],'f1List2' : f1List2[:10], 'f2List2' : f2List2[:10],
#                     'name1' : name1[:10], 'name2' : name2[:10],})
#
# print(df)

#
# with open("data/synsetDict1200.pickle", "rb") as fr:
#     dict1200 = pickle.load(fr)
#
# with open("data/synsetDict_1000.pickle", "rb") as fr:
#     dict1000 = pickle.load(fr)
#
#
#
# n1200 = dict1200.values()
# n1000 = dict1000.values()
#
# cnt1200 = Counter(n1200)
# cnt1000 = Counter(n1000)
#
# print('dict1200 : ',len(list(set(n1200))))
# print('dict1000 : ',len(list(set(n1000))))
#
# print('cnt1200 : ',len(cnt1200))
# print('cnt1000 : ',len(cnt1000))
#
# print('cnt1200_mostCommon : ',cnt1200.most_common(20))
# print('cnt1000_mostCommon : ',cnt1000.most_common(20))
