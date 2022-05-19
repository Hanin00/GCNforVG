import sys

import numpy as np
import pandas as pd
import util as ut
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import pickle

'''
    Ex edge per 1Img
'''
# objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(1)
# df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, "edgeId": edgeId,
#        "weight": weight, "relatiohship": relatiohship, })
# print(df_edge.head(5))

''' 
    Ex node per 1Img
'''
# objectIds, objectNames = ut.AllNodes(1)
# df_node = pd.DataFrame({"objectIds": objectIds, "objectNames": objectNames})
# print(df_node.head(5))


'''
    feature Matrix list(1000,len(objectName),10) 형태로 저장하기
'''
# imgCnt = 1001
# features = []
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
#
# # fasttext 240마다 멈춰서
# for j in tqdm(range(1,imgCnt)):
#     objectIds, objectNames = ut.AllNodes(data, j)
#     objDict = {name: value for name, value in zip(objectIds, objectNames)}
#     embDict = ut.FeatEmbeddPerImg(objectNames)
#     features.append(ut.FeatureMatrix(objectIds, objDict, embDict))
#
# featM = np.array(features)
# with open("./data/featMatrix1000.pickle", "wb") as fw:
#     pickle.dump(featM, fw)
#
# with open("./data/featMatrix1000.pickle", "rb") as fr:
#     data = pickle.load(fr)

# print(data.size)
# print(data[0].shape)
# print(type(data[0]))

'''
    NetworkX 객체 리스트 생성 - 1000개의 이미지에 대한 ObjId, SubjId의 realationship에 대한 Graph 객체 생성
    -> 총 1000개의 undirected Graph
    size (1000, [ObjId 개수])
    >> AddEdges 사용
'''
# gList = []
# imgCnt = 1000
# start = time.time()
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
# end = time.time()
# print(f"파일 읽는데 걸리는 시간 : {end - start:.5f} sec") # 파일 읽는데 걸리는 시간 : 24.51298 sec
#
# for i in tqdm(range(imgCnt)):
#     objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(i,data)
#     df_edge = pd.DataFrame({"objId": objId, "subjId": subjId,})
#     gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')
#     gList.append(gI)
#
# with open("./data/neworkx1000.pickle", "wb") as fw:
#     pickle.dump(gList, fw)

# with open("./data/neworkx1000.pickle", "rb") as fr:
#     data = pickle.load(fr)

# G = data[1]
# print(G.nodes.data())


''' NetworkX 객체에서 Adj 추출하기 - nodeList option으로 줘서 순서에 맞게!!'''
# 굳이 할 필요 없이

# with open("./data/neworkx1000.pickle", "rb") as fr:
#     data2 = pickle.load(fr)

# G = data2[0]
# nx.draw(G, with_labels=True, font_size=5)
# pos = nx.spring_layout(G)
# plt.show()


''' Feature Matrix local 저장 및 Adj local 저장'''
with open('./data/scene_graphs.json') as file:  # open json file
    data1 = json.load(file)
    # objId, objName 불러옴
errorId = [] # todo relationship에서 Id를 얻어오지 못하는 것을 확인함 -> 해당 Id 기록해서 체크해보기

featMList = []
adjMList = []
imgCnt = 1300
for imgId in tqdm(range(50, imgCnt)):
    objectIds, objectNames = ut.AllNodes(data1, imgId)
    # objectId는 중복 X, ordered는 아님

    objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data1, imgId)
    objIdList = []
    objIdList += objId
    objIdList += subjId
    objIdList = list(set(objIdList))

    objDict = {name: value for name, value in zip(objectIds, objectNames)}
    embDict = ut.FeatEmbeddPerImg(objectNames)
    featM = ut.FeatureMatrix(objIdList, objDict, embDict)
    featMList.append((featM))

    df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })
    if(len(df_edge) == 0) :
        errorId.append(imgId)
        continue
    else :
        G = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')
        A = nx.adjacency_matrix(G, objIdList)  #
        # A1 = A.todense()
        adjMList.append(A)

print(errorId)

with open("./data/featMatrix1000.pickle", "wb") as fw:
    pickle.dump(featMList, fw)
with open("./data/adjMatrix1000.pickle", "wb") as fw:
    pickle.dump(adjMList, fw)