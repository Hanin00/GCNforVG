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

'''
    1. 1000개, 200개 동일한 특징 노드에 클래스의 특징이 같은 지 확인
    2. 1000개 senset과 1200개 까지의 synset에서 다른 게 몇 개 인지?
'''
with open("data/networkx_ver2.pickle", "rb") as fr:
    netv2G = pickle.load(fr)

with open("data/networkx_ver200.pickle", "rb") as fr:
    n200G = pickle.load(fr)


for i in range(len(netv2G)) :
    names = [row[1] for row in netv2G[i].nodes(data='name')]
    if 'grass' in names :
        g1Id = i
        break

for i in range(len(n200G)) :
    names = [row[1] for row in n200G[i].nodes(data='name')]
    if 'grass' in names :
        g2Id = i
        break


G1 = netv2G[g1Id]
G2 = n200G[g2Id]

f0List1 = [row[1] for row in G1.nodes(data='f0')]
f1List1 = [row[1] for row in G1.nodes(data='f1')]
f2List1 = [row[1] for row in G1.nodes(data='f2')]
name1 = [row[1] for row in G1.nodes(data='name')]

f0List2 = [row[1] for row in G2.nodes(data='f0')]
f1List2 = [row[1] for row in G2.nodes(data='f1')]
f2List2 = [row[1] for row in G2.nodes(data='f2')]
name2 = [row[1] for row in G2.nodes(data='name')]








pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)

df = pd.DataFrame({ 'f0List1' : f0List1[:10],'f1List1' : f1List1[:10], 'f2List1' : f2List1[:10],
                    'f0List2' : f0List2[:10],'f1List2' : f1List2[:10], 'f2List2' : f2List2[:10],
                    'name1' : name1[:10], 'name2' : name2[:10],})

print(df)



with open("data/synsetDict1200.pickle", "rb") as fr:
    dict1200 = pickle.load(fr)

with open("data/synsetDict_1000.pickle", "rb") as fr:
    dict1000 = pickle.load(fr)


cnt1200 = Counter(dict1200)
cnt1000 = Counter(dict1000)

print('dict1200 : ',len(dict1200))
print('dict1000 : ',len(dict1000))

print('cnt1200 : ',len(cnt1200))
print('cnt1200 : ',len(cnt1200))

print('cnt1200_mostCommon : ',cnt1200.most_common(10))
print('cnt1000_mostCommon : ',cnt1000.most_common(10))