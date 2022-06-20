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
from collections import Counter

'''
    objId를 기준으로 그래프를 생성하되, 
    1. 단일 node에 이웃 노드 이름이 5개 이상 겹치는 경우, 이름이 같은 오브젝트들을 모두 하나의 node로 묶고 오브젝트를 삭제함
    2. neighbor node의 범위가 중복되는 경우 id를 name 기준으로 합친 후 묶인 id 중에서 제일 첫번째의 id부여하고 (근처에 위치하는) 동일 이름을 가진 obj의 id값을 새로 부여한 id로 치환
    -> 첫번째의 id 값으로 node id를 모두 변경하는 List를 새로 만들어서, 
    3. 중요 object Name은 합치지 않도록 분기처리 할 것
    
    2-a : 애초에 하나의 노드에 이웃하는 노드들이 동일한 name을 갖는 경우 하나의 id로 통일할 거니까 상관 없는거 아닌가? x, y값 필요 없지 않나?
    -> 나중에 vizualize 할 때만 필요. 이때에도 단순히 새로 생성한 id들의 집합을 {새로 생성한 id : [묶인 ids]} 형태로 저장했다가 visualize 할 때만 보면 될 듯

'''


def get_key(dict, val):
    for key, value in dict.items():
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

objNamesList = []
for imgId in tqdm(range(imgCnt)):
    objectIds, objectNames = ut.AllNodes(data, imgId)
    objNamesList += objectNames
objNamesList = list(set(objNamesList))
totalEmbDict = ut.FeatEmbeddPerTotal(objNamesList)

for i in range(imgCnt):
    objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data, i)
    # networkX graph 객체 생성 ---
    objIdSet, objNameList = ut.AllNodes(data, i)
    df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })
    gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')
    nodesList = sorted(list(gI.nodes))

    # node attribute 부여 ---
    embDict = ut.MatchDictImage(objIdSet, objNameList, totalEmbDict)

    neighUpp5 = []
    for nodeId in nodesList:  # nodeId
        # neighbors = gI.neighbors(nodeId)
        if (len(list(gI.neighbors(nodeId)))) >= 5:
            neighUpp5.append(nodeId)
        # neighbors 5개 이상인 것들의 nodeId

    idToName = {id: name for id, name in zip(objIdSet, objNameList)}

    '''
    Neighbors의 objectName 확인, 5개 이상 동일한 경우, 해당 Neighbor의 Id를 묶고, sort 함.
    이후 전체 ObjId List에서 바꿔줌. Id로 이름 호출. get_key 사용해서 이름으로 Id 호출
    '''

    # 전체 노드 id에 대해 변경해야 할 Id List fId = 리스트들에서 제일 작은 Id, totalList = [[아이디들] , []],nameList = [동일한 ObjName] <- 예외처리를 위해
    fId = []
    totalList = []
    nameList = []

    for nodeId in neighUpp5:
        neighbors = list(gI.neighbors(nodeId))
        neiNames = [idToName[k] for k in neighbors]
        print('list(neiNames) : ', list(neiNames))

        sameName = list(Counter(neiNames).keys())
        sameNums = list(Counter(neiNames).values())
        sameUpp5 = list(filter(lambda num: num >= 5, sameNums))

        #todo 분기처리 - 예외 단어 추가하기
        exceptionalWords = []

        a = []
        if sameUpp5 != 0:
            for i in sameUpp5:
                #todo 분기처리 - 여기서 고려대상 나오면 걍 넘기기
                if sameName[sameNums.index(i)] in exceptionalWords :
                    continue
                else :
                    a.append(sameName[sameNums.index(i)])

        if len(a) != 0:
            b = []
            for name in a:
                for key, value in idToName.items():
                    if name == value:
                        b.append(key)
            if len(b) != 0:
                b = sorted(b)
                fId.append(b[0])
                totalList.append(b)
                nameList += a

    # 동일한 이름이 있을 때 nameList의 원소가 동일한 것들의 nameList.index() 구해서,
    # totalList(idx) 끼리 더하고, sort 해서 fId
    # if len(fId) != len(nameList) :

    # 하나에 너무 많이 겹치는 거 아닌가 그러면?

    # dictionary = totalList[i][j] : fId[i]

    replaceDict = {}
    for i in range(len(totalList)):
        for j in range(len(totalList[i])):
            replaceDict[str(totalList[i][j])] = fId[i]

    print('replace : ', replaceDict)

    newObjList = []
    newSubjList = []
    if(len(replaceDict) != 0) :
        for i in objId:
            try:
                newObjList.append(replaceDict[str(i)])
                print('replaceDict[str(i)] : ',replaceDict[str(i)])
                print(i,'를 ', replaceDict[str(i)],'로')
            except KeyError:
                newObjList.append(i)

        for i in subjId:
            try:
                newSubjList.append(replaceDict[str(i)])
            except KeyError:
                newSubjList.append(i)

        objId, subjId = newObjList, newSubjList
        df_new = pd.DataFrame({"objId": objId, "subjId": subjId, })
        gI = nx.from_pandas_edgelist(df_new, source='objId', target='subjId')
        nodesList = sorted(list(gI.nodes))

    for i in range(len(nodesList)):  # nodeId
        nodeId = nodesList[i]
        emb = embDict[nodeId]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
        for j in range(3):  # Embedding 값은 [3,]인데, 각 원소를 특징으로 node에 할당
            nx.set_node_attributes(gI, {nodeId: emb[j]}, "f" + str(j))

    # graph에서 노드 id 0부터 시작하도록 ---
    listA = list(set(objId + subjId))
    listIdx = range(len(listA))
    dictIdx = {name: value for name, value in zip(listA, listIdx)}
    gI = nx.relabel_nodes(gI, dictIdx)
    #  nx.set_node_attributes(gI, 1, "weight")
    gList.append(gI)

with open("./data/networkx_sifted.pickle", "wb") as fw:  # < node[nId]['attr'] = array(float)
    pickle.dump(gList, fw)

with open("./data/networkx_sifted.pickle", "rb") as fr:
    data = pickle.load(fr)




gId = 294
gI = data[gId]
print(data)
print('data[gId] : ', data[gId])





plt.figure(figsize=[15, 7])
nx.draw(gI, with_labels=True)
plt.show()
