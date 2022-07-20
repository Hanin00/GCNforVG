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



def graphShowName(nexG):
    #nx.draw(nexG, with_labels=True)
    relabelDict = {}
    for idx in nexG.nodes():
        relabelDict[idx] = nexG.nodes[idx]['name']

    nexG = nx.relabel_nodes(nexG, relabelDict)
    plt.figure(figsize=[4, 4])
    nx.draw(nexG,  with_labels=True)
    plt.show()

def vGphShow(nexG):
    plt.figure(figsize=[15, 7])
    nx.draw(nexG, with_labels=True)
    plt.show()

# with open("data/networkx_ver2.pickle", "rb") as fr:
#     ver2G = pickle.load(fr)

def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

#synset 데이터에서 가장 많이 언급된 단어, 적게 언급된 단어
# with open("data/v3_x100/synsetDictV3_x100.pickle", "rb") as fr:
#     synsetDict = pickle.load(fr)
#
# pd.set_option('display.max_rows', None)
#
# synCnt = Counter(synsetDict.values())
# # vals = [i[0] for i in synCnt.most_common(1000)]
# # valcnt = [i[1] for i in synCnt.most_common(1000)]
# vals = [i[0] for i in synCnt.most_common()]
# valcnt = [i[1] for i in synCnt.most_common()]
# df = pd.DataFrame(vals, valcnt)
# print(df.tail(20000))
#
# sys.exit()

# # 단어에 따른 graph 번호
# def mkCnt(imgCnt,data, name) :
#     nodeNameList = []
#     for gId in range(imgCnt) :
#         nodeList = data[gId].nodes(data = 'name')
#         nodeNameList += [node[1] for node in nodeList]
#         if name in nodeNameList :
#             return gId
#
# with open("data/v3_x100/v3_x1000.pickle", "rb") as fr:
#     v3X1000 = pickle.load(fr)
#
# name = 'hand_blower'
# gId = mkCnt(len(v3X1000), v3X1000, name)
# graphShowName(v3X1000[gId])
#
# sys.exit()


with open("data/v3_x100/v3_x1000.pickle", "rb") as fr:
    v3X1000 = pickle.load(fr)

with open("data/v3_x100/totalEmbDictV3_x100.pickle", "rb") as fr:
    embDict= pickle.load(fr)



nodeNameList = [
                [['table','table','table','table'],
                  ['vase','flower','chair','cup']] ,

                [['table','vase','table','table','table'],
                  ['vase','flower','flower','fabric','cup']] ,

                [['car','car','car','car','window'],
                  ['light','wheel','window','tire','building']],

                 [['car', 'car', 'car',  'car'],
                  ['tire', 'roof', 'pipe' , 'garage']],

                 [[ 'building', 'window', 'sign', 'building',],
                  ['window', 'sign', 'letter', 'light']],

                 [['display', 'display', 'water', 'animal'],
                  ['palm', 'water', 'animal', 'fence']],

                 [['animal', 'animal', 'animal', 'fence', 'animal'],
                  ['platform', 'fence', 'spectator', 'spectator', 'soil' ]],

                 [['table', 'table', 'lamp', 'flower'],
                  ['flower', 'lamp', 'cabinet', 'vase',]],

                [['street', 'street', 'street', 'car'],
                    ['line', 'shadow', 'car', 'headlight', ]],

                [['man', 'man', 'car', 'car'],
                    ['cap', 'car', 'person', 'engine',]],

                [['man', 'man', 'vehicle', 'man', 'man'],
                 ['vehicle', 'motor', 'motor', 'jean', 'shirt' ]],

                [['ear', 'ear', 'man', 'man', 'car'],
                 ['man', 'hat', 'hat', 'car', 'engine' ]],

                [['lampshade', 'lamp', 'lamp', 'shadow'],
                 ['lamp', 'table', 'shadow', 'wall', ]],

                [['wall', 'wall', 'wall', 'painting'],
                 ['picture', 'painting', 'shadow', 'frame', ]],

                 [['bed', 'bed', 'bed', 'bed'],
                  ['bench', 'blouse', 'pillow', 'sheet', ]],

                [['bed',  'pillow','table', 'table'],
                 ['pillow', 'table', 'lamp', 'mirror', ]],


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


with open("./data/query01_0720_2.pickle", "wb") as fw:
    pickle.dump(gList, fw)

with open("./data/query01_0720_2.pickle", "rb") as fr:
    gList = pickle.load(fr)

print(gList[0].nodes(data=True))
print(gList[1].nodes(data=True))
print(gList[2].nodes(data=True))

[graphShowName(graph) for graph in gList]
#graphShowName(gList[2])
