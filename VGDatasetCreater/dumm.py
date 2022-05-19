import pandas as pd
import util as ut
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time
import json
import pickle
from gensim.models import FastText
from tqdm import tqdm

objectNames = ['string','string2']

objectNames += ['road', 'taxi', 'door', 'taxi', 'wagon', 'man', 'stairs', 'suit', 'street', 'van', 'sign', 'car',
                'symbol', 'car', 'trafficcone', 'ground', 'tag', 'lights', 'woman', 'cars', 'people', 'car', 'people',
                'steps', ]

print(objectNames)

# gList = []
# imgCnt = 3
#
# start = time.time()
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
# end = time.time()
# print(f"파일 읽는데 걸리는 시간 : {end - start:.5f} sec")
#
#
# for i in tqdm(range(imgCnt)):
#     start = time.time()
#     objId, subjId, relatiohship, edgeId, weight = ut.AllEdges2(i,data)
#     df_edge = pd.DataFrame({"objId": objId, "subjId": subjId,})
#     gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')
#     gList.append(gI)
#     end = time.time()
#     print(f"for 문 한 번에 : {end - start:.5f} sec")
#
# #AllEdge : for 문 한 번에 : 27.80394 sec
# #AllEdge2 : for 문 한 번에 : 0.00100 sec
