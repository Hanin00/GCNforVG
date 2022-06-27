import sys
import numpy as np
import util as ut
import json
import pickle
import networkx as nx
from collections import Counter
from nltk.corpus import conll2000
from tqdm import tqdm
import pandas as pd
from visual_genome import api as vg
import matplotlib.pyplot as plt

'''
    synsetDict는 scene_graph.json의 objects를 기반으로 추출함
    graph는 relationship 기반으로 생성함
    relationship은 이미지의 모든 objects를 모두 사용하진 않음
    -> synsetDict.valuse() 의 중복 제거한 값과 실제 생성한 graph의 노드 name 개수는 다름

'''

with open("data/test_100.pickle", "rb") as fr:
    data = pickle.load(fr)

print(len(data[0].nodes()))
print(len(data[0].nodes()))


imgCnt = 100

def mkCnt(imgCnt,data ) :
    nodeNameList = []
    for i in range(imgCnt) :
        nodeList = data[i].nodes(data = 'name')
        nodeNameList += [node[1] for node in nodeList]

    synsetCnt = Counter(nodeNameList)
    print(len(list(set(nodeNameList))))
    return synsetCnt


synsetCnt = mkCnt(imgCnt, data)

print(synsetCnt['Ground'])
print(synsetCnt['paper'])
lista = list(synsetCnt.most_common())
print(len(lista))


# with open("data/nx_v2_classinfo_10000.pickle", "wb") as fw:  # < node[nId]['attr'] = array(float)
#     pickle.dump(synsetCnt, fw)
