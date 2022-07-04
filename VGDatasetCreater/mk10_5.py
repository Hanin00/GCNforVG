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
import time

'''
    100000개짜리 synset 생기면 사용해서 10000개씩 저장
'''

with open("data/synsetDictV3_x100.pickle", "rb") as fr:
    synsDict = pickle.load(fr)

with open("data/totalEmbDictV3_x100.pickle", "rb") as fr:
    totalEmbDict = pickle.load(fr)

with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)

lista = list(range(100000))


for i in range(10) :
    names = []
    start = i
    if i !=0:
        start = i*10000
    end = (i+1)*10000
    print(start," : ", end)

    gList = []
    for j in lista[start:end] :
        objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data, i)
        objIdSet, objNameList = ut.AllNodes(data, i)

        # 이름이 중복되면 value 값 갱신됨
        # 이름 하나에 하나의 i 값만 갖는 dict
        # idNameDict = {ObjIdx : SynsetName} 를 만든기 위해 전체 objIdList를 사용

        attrNameList = []  # name attr  추가를 위한 코드
        for kId in objIdSet:
            attrNameList.append(synsDict[str(kId)])
        # idNameDict = {이미지 내 ObjIdx : NameList}
        idNameDict = {str(idx): synsetName for idx, synsetName in zip(objIdSet, attrNameList)}  # name attr  추가를 위한 코드

        newObjName = []  # name attr  추가를 위한 코드
        newSubjName = []  # name attr  추가를 위한 코드

        # Obj,Subj의 Id에 해당하는 SynsetName List 생성
        for oId in objId:
            try:
                newObjName.append(idNameDict[str(oId)])
            except:
                newObjName.append('')

        for sId in subjId:
            try:
                newSubjName.append(idNameDict[str(sId)])
            except:
                newSubjName.append('')

        # 자기 자신에 참조하는 node가 있는 Idx List 생성
        recurRowId = []
        for j in range(len(objId)):
            if objId[j] == subjId[j]:
                recurRowId.append(j)

        df_edge = pd.DataFrame(
            {"objId": objId, "subjId": subjId, "newObjName": newObjName, "newSubjName": newSubjName, })

        # 자기 자신을 참조하는 노드의 relationship 삭제
        if recurRowId != 0:
            for idx in recurRowId:
                df_edge = df_edge.drop(index=idx)
        df_edge = df_edge.replace(r'', np.nan, regex=True)
        df_edge = df_edge.dropna()

        gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')

        # --------- ^^^ Graph 생성, graph에 name, Origin ObjId를 attribute로 추가함 ^^^ ------------------
        #                   자기 자신 참조하는 중복 제거, synset name 적용

        # ----------- vvv 이웃 노드에 동일한 이름을 가진 노드가 5개 이상인 경우 동일 id로 변환 vvv -------------

        nodesList = sorted(list(gI.nodes))
        objIdSet = df_edge['objId'].tolist() + df_edge['subjId'].tolist()
        objNameList = df_edge['newObjName'].tolist() + df_edge['newSubjName'].tolist()

        # 위에서 제거된 값을 위해 변경
        objId = df_edge['objId'].tolist()
        subjId = df_edge['subjId'].tolist()

        neighUpp5 = []  # neighbors 5개 이상인 것들의 nodeId
        for nodeId in nodesList:
            if (len(list(gI.neighbors(nodeId)))) >= 5:
                neighUpp5.append(nodeId)

        '''
        Neighbors의 objectName 확인, 5개 이상 동일한 경우, 해당 Neighbor의 Id를 묶고, sort 함.
        이후 전체 ObjId List에서 바꿔줌. Id로 이름 호출. get_key 사용해서 이름으로 Id 호출
        '''

        # 전체 노드 id에 대해 변경해야 할 Id List fId = 리스트들에서 제일 작은 Id, totalList = [[아이디들] , []],nameList = [동일한 ObjName] <- 예외처리를 위해
        fId = []
        totalList = []

        for nodeId in neighUpp5:
            neighbors = list(gI.neighbors(nodeId))
            neiNames = [idNameDict[str(k)] for k in neighbors]

            sameName = list(Counter(neiNames).keys())
            sameNums = list(Counter(neiNames).values())
            sameUpp5Idx = []
            for idx in range(len(sameNums)):
                if sameNums[idx] >= 5:
                    sameUpp5Idx.append(idx)

            # delTargetnames : 한 노드의 이웃노드면서 5개 이상 중복되는 objectName List
            delTargetNames = []

            # todo 분기처리 - 예외 단어 추가하기 / 예외단어 : 중복이어도 허용하는 important Name
            exceptionalWords = []
            if sameUpp5Idx != 0:
                for idx in sameUpp5Idx:
                    # todo 분기처리 - 여기서 고려대상 나오면 걍 넘기기
                    if sameName[idx] in exceptionalWords:
                        continue
                    else:
                        delTargetNames.append(sameName[idx])

            if len(delTargetNames) != 0:
                delTargetIds = []
                for name in delTargetNames:
                    for key, value in idNameDict.items():
                        if name == value:
                            delTargetIds.append(key)

                delTargetIds = sorted(delTargetIds)
                fId.append(delTargetIds[0])
                totalList.append(delTargetIds)

        # todo 앞서 변경된 Id가 나중에 변경된 Id 값에 의해 왕창 늘어날 가능성 고려 및 코드 수정 필요
        # replaceDict = {delTargetIds : delTargetIds 중 변경 대상이 되는 Id}
        replaceDict = {}
        for idx in range(len(totalList)):
            for jIdx in range(len(totalList[idx])):
                replaceDict[str(totalList[idx][jIdx])] = fId[idx]

        newObjList = []
        newSubjList = []
        if (len(replaceDict) != 0):
            for idx in objId:
                try:
                    newObjList.append(replaceDict[str(idx)])  # 교체 대상인 경우 교체
                except KeyError:
                    newObjList.append(str(idx))  # 이웃 노드에서 중복이 아니어서 교체 대상이 아닌 경우 기존 Id

            for idx in subjId:
                try:
                    newSubjList.append(replaceDict[str(idx)])  # 교체 대상인 경우 교체
                except KeyError:
                    newSubjList.append(str(idx))  # 이웃 노드에서 중복이 아니어서 교체 대상이 아닌 경우 기존 Id

        if len(newObjList) != 0:
            newObjId, newSubjId = newObjList, newSubjList
        else:
            newObjId, newSubjId = objId, subjId

        # id 값으로 name 호출 - obj / subj -> idtoName
        newObjName = [idNameDict[str(idx)] for idx in newObjId]
        newSubjName = [idNameDict[str(idx)] for idx in newSubjId]

        df_new = pd.DataFrame({"objId": newObjId, "subjId": newSubjId,
                               "objName": newObjName, "subjName": newSubjName})

        df_new['objId'] = df_new['objId'].astype(int)
        df_new['subjId'] = df_new['subjId'].astype(int)

        objIdSet = df_new['objId'].tolist() + df_new['subjId'].tolist()

        gI = nx.from_pandas_edgelist(df_new, source='objId', target='subjId')
        for index, row in df_new.iterrows():
            gI.nodes[row['objId']]['name'] = row["objName"]  # name attr
            gI.nodes[row['subjId']]['name'] = row['subjName']  # name attr

            gI.nodes[row['objId']]['originId'] = row['objId']  # originId attr
            gI.nodes[row['subjId']]['originId'] = row['subjId']  # originId attr

        nodesList = sorted(list(gI.nodes))
        embList = [totalEmbDict[synsDict[str(idx)]] for idx in nodesList]
        embDict = {idx: emb for idx, emb in zip(nodesList, embList)}

        for idx in range(len(nodesList)):  # nodeId
            nodeId = nodesList[idx]
            emb = embDict[nodeId]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
            nx.set_node_attributes(gI, {nodeId: float(emb)}, "f0")
        # node relabel - graph에서 노드 id 0부터 시작하도록 ---
        dictIdx = {nodeId: idx for idx, nodeId in enumerate(nodesList)}
        gI = nx.relabel_nodes(gI, dictIdx)
        gList.append(gI)

    path = "./data/v3_x100/v3_x100" + str(i) + ".pickle"
    with open(path, "wb") as fw:
        pickle.dump(gList, fw)

with open("data/v3_x100/v3_x1001.pickle", "rb") as fr:
    graphs= pickle.load(fr)

print(graphs[0])
