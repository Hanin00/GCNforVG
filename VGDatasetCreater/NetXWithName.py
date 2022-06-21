import sys
import numpy as np
import pandas as pd
import torch
import csv

import torch_geometric.utils

import util as ut
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import pickle

'''
    0617일자 id 기반으로 그래프 생성 후 Object Name 기준 relabel 
     -> relabel 된 node id 기반으로 그래프 생성 
'''


def get_key(val):
    for key, value in idIdxDict.items():
        if val == value:
            return key
    return "key doesn't exist"

gList = []
imgCnt = 1000
start = time.time()
with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
end = time.time()
print(f"파일 읽는데 걸리는 시간 : {end - start:.5f} sec")  # 파일 읽는데 걸리는 시간 : 24.51298 sec



imgCnt = 10
synsetList = []
nonSysnsetList = []

d = data[24]["objects"]

for ImgId in range(imgCnt) :
    imageDescriptions = data[ImgId]["objects"]
    print("len(imageDescriptions : ",len(imageDescriptions))
    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        try :
            a = imageDescriptions[j]['synsets'][0].split(".")
            synsetList.append(a[0])
        except Exception  :
            nonSysnsetList.append(imageDescriptions[j]['names'][0])
            print(Exception)
            print("ImgId : ", ImgId)
            print('j : ', j)



synsetList = list(set(synsetList))
nonSysnsetList = list(set(nonSysnsetList))

print("synsetList : ", synsetList)
print("nonSysnsetList : ", nonSysnsetList)
print(len(synsetList))
print(len(nonSysnsetList))







sys.exit()






objNamesList = []
for imgId in tqdm(range(imgCnt)):
    objectIds, objectNames = ut.AllNodes(data, imgId)
    objNamesList += objectNames
objNamesList = list(set(objNamesList))

totalEmbDict = ut.FeatEmbeddPerTotal(objNamesList)
with open("./data/totalEmbDict.pickle", "wb") as fw:
    pickle.dump(totalEmbDict, fw)

# with open("./data/totalEmbDict.pickle", "rb") as fr:
#     data = pickle.load(fr)
# totalEmbDict = data

for i in tqdm(range(imgCnt)):
    objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data, i)
    # networkX graph 객체 생성 ---
    objIdSet, objNameList = ut.AllNodes(data, i)
    # df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })

    ''' 
        Object Name List를 기준으로 ReLabeling(String -> Int)
        Node List 생성
        1. ObjNameList의 
            relabelDict - {name : newIdx}
        2. newObjNameList = []
        2. idIdxDict = {ObjIdSet : relabelDict[objIdSet]}
        3. newObjId = idIdxDict[objId(i)]
           newSubjId = idIdxDict[subjId(i)]

           이름에 대한 NodeList
    '''
    # 이름이 중복되면 value 값 갱신됨
    # 이름 하나에 하나의 i 값만 갖는 dict
    relabelDict = {objName: i for i, objName in enumerate(objNameList)}

    newObjIdList = []
    attrNameList = []    #name attr  추가를 위한 코드
    for kId in range(len(objIdSet)):
        attrNameList.append(objNameList[kId])
        newObjIdList.append(relabelDict[objNameList[kId]])        # name attr  추가를 위한 코드
    idIdxDict = {name: value for name, value in zip(objIdSet, newObjIdList)}
    idNameDict = {name: value for name, value in zip(objIdSet, attrNameList)}    #name attr  추가를 위한 코드

    newObjId = []
    newSubjId = []

    newObjName = []       #name attr  추가를 위한 코드
    newSubjName = []      #name attr  추가를 위한 코드
    for oId in objId:
        newObjId.append(idIdxDict[oId])
        newObjName.append(idNameDict[oId])

    for sId in subjId:
        newSubjId.append(idIdxDict[sId])
        newSubjName.append(idNameDict[sId])

    recurRowId = []
    for j in range(len(newObjId)):
        if newObjId[j] == newSubjId[j]:
            recurRowId.append(j)

    embDict = ut.MatchDictImage(objIdSet, objNameList, totalEmbDict)
    df_edge = pd.DataFrame({"objId": newObjId, "subjId": newSubjId,"newObjName": newObjName, "newSubjName": newSubjName, })
    #df_edge = pd.DataFrame({"objId": newObjId, "subjId": newSubjId, })
    #df_name = pd.DataFrame({"newObjName": newObjName, "newSubjName": newSubjName, })

    if recurRowId != 0:
        for idx in recurRowId:
            df_edge = df_edge.drop(index=idx)
            #df_name = df_name.drop(index=idx)

    gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')
    for index, row in df_edge.iterrows():
        gI.nodes[row['objId']]['name'] = row['newObjName']
        gI.nodes[row['subjId']]['name'] = row['newSubjName']

    nodesList = list(gI.nodes)

    for nodeId in nodesList:  # nodeId
        '''
            idIdxDict에서 nodeId가 value인 key를 찾으면 원래의 ObjectId 가 나옴,
            원래의 Object Id로 Name을 찾음
            Name을 토대로 totalEmbDict의 value를 호출
        '''
        originObjId = get_key(nodeId)
        nx.set_node_attributes(gI, originObjId, "originId")
        emb = embDict[originObjId]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
        for j in range(3):  # Embedding 값은 [3, ]인데, 각 원소를 특징으로 node에 할당
            nx.set_node_attributes(gI, {nodeId: emb[j]}, "f" + str(j))

    # graph에서 노드 id 0부터 시작하도록 ---
    listA = list(set(newObjId + newSubjId))
    listIdx = range(len(listA))
    dictIdx = {name: value for name, value in zip(listA, listIdx)}
    gI = nx.relabel_nodes(gI, dictIdx)
    gList.append(gI)


with open("./data/networkx_synset.pickle", "wb") as fw:  # < node[nId]['attr'] = array(float)
    pickle.dump(gList, fw)
    # pickle.dump(gList[:1000], fw)

with open("./data/networkx_synset.pickle", "rb") as fr:
    data = pickle.load(fr)

gId = 294
gI = data[gId]
print(data)
print('data[gId] : ', data[gId])
print('data[gId].nodes() : ', data[gId].nodes(data=True))

plt.figure(figsize=[15, 7])
nx.draw(gI, with_labels=True)
plt.show()
