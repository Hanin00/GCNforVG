import sys

import pandas as pd
from VGDataTrustValidation import util as ut
import networkx as nx
import json
import pickle


# Graph A의 id를 Graph B의 node attr ['originId'] 로.
with open("data/networkx_sifted_name.pickle", "rb") as fr:
    data1 = pickle.load(fr)

with open("data/networkx_sifted_oIdx.pickle", "rb") as fr:
    data2 = pickle.load(fr)


targetGList = data1

sorIdGList = data2

nGList = []
originList =[]

for g in sorIdGList :
    originList.append(list(g.nodes()))


for g in range(len(targetGList)) :
    graph = targetGList[g]
    nodes = list(graph.nodes())
    originL = originList[g]
    kDict = {ns: originIdx for ns, originIdx in zip(nodes, originL)}

    for j in list(graph.nodes()) :
        graph.nodes[j]["originId"] = kDict[j]
    nGList.append(graph)

with open("data/networkx_ver1.pickle", "wb") as fw:  # < node[nId]['attr'] = array(float)
    pickle.dump(nGList, fw)

with open("data/networkx_ver1.pickle", "rb") as fr:
    data = pickle.load(fr)

print(data[0].nodes(data=True))
