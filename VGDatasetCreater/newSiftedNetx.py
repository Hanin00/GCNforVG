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
import nltk
from nltk.corpus import conll2000

# todo - 0부터 시작하는 값으로 노드의 라벨을 변경하는 작업을 가장 마지막에 해야하는데
# 먼저 변경하는 바람에 기존 originId를 이용해 Object의 위치를 알 수 없는 문제 발생 -> 해당 오류 정정 필요


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


def blank_nan(x):
    if x == '':
        x = np.nan
    return x


'''
    Synset Naming
    2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
        -> noun이 두 개 일 경우 기존 synset List에서 가장 많이 사용되는 단어를 사용하고, 개수가 일치 할 경우 [0]의 단어를 name으로 함
        -> NLTK로 분할한 noun들이 기존 synset이 아닌 경우 전체를 통째로 synset으로 만들기
'''


def extractNoun(noun, synsDict, synNameCnter):
    conllDict = {word: tag for word, tag in conll2000.tagged_words(tagset='universal')}

    words = noun.split(' ')
    # noun 판별
    nouns = []
    for i in words:
        try:
            if conllDict[i] == 'NOUN':
                nouns.append(i)
        except:
            continue
    # synset에 해당되는 noun이 있는지 판별
    nInSynsDictList = []
    if len(nouns) != 0:
        for i in nouns:
            try:
                nInSynsDictList.append(synsDict[i])
            except:
                continue

    # synset에 해당되는 noun 중 언급이 많은 단어 선택
    name = ''
    cnt = 0
    if len(nInSynsDictList) != 0:
        for i in nInSynsDictList:
            if cnt < synNameCnter[i]:
                name = i
                cnt = synNameCnter[i]
    else:
        name = "_".join(sorted(nouns))
    return name


gList = []
imgCnt = 1000

with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
end = time.time()

# --------------------------- vvv synset Dict, Total Embedding(fasttext 값) vvv ---------------------------
'''synset Name Dict'''

synsetList = []
originIdList = []
nonSysnNameList = []
nonSysnIdList = []
originDict = {}

for ImgId in range(imgCnt):
    imageDescriptions = data[ImgId]["objects"]
    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        oId = imageDescriptions[j]['object_id']
        try:
            synsetName = imageDescriptions[j]['synsets'][0].split(".")
            synsetList.append(synsetName[0])
            originIdList.append(str(oId))
            originDict[str(oId)] = imageDescriptions[j]['names'][0]

        except Exception:
            nonSysnNameList.append(imageDescriptions[j]['names'][0])
            nonSysnIdList.append(str(oId))

synNameCnter = Counter(synsetList)
'''   
    Synset Naming
    1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우, objId로 synsDict에서 synsetName을 찾음
    2. nonSynsetName/Id에서 해당 원소를 제외하고, synsDict에 추가함(objId : objName(synset))

    없는 경우, 2로 넘어감
'''

synsDict = {idx: name for idx, name in zip(originIdList, synsetList)}
nonSynsDict = {name: value for name, value in zip(nonSysnIdList, nonSysnNameList)}

for i in range(len(nonSysnNameList)):
    try:
        sameNameId = get_key(originDict, nonSysnNameList[
            i])  # 1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우,
        synsDict[str(nonSysnIdList[i])] = synsDict[sameNameId]

    except:
        # todo 2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
        name = extractNoun(nonSysnNameList[i], synsDict, synNameCnter)
        synsDict[str(nonSysnIdList[i])] = name

# #위에서 만든 synset Dict를 이용해 totalEmbedding 값을 만듦(fasttext)
objectNameList = list(set(list(synsDict.values())))
model, totalEmbDict = ut.FeatEmbeddPerTotal_model(objectNameList)
with open("./data/totalEmbDict.pickle", "wb") as fw:
    pickle.dump(model, fw)
# --------------------------- ^^^ synset Dict, Total Embedding(fasttext 값)^^^ ---------------------------

for i in tqdm(range(imgCnt)):
    objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data, i)
    objIdSet, objNameList = ut.AllNodes(data, i)

    objIdList = []
    objNameListSynset = []
    for id in objIdSet:  # relationship으로 부터 반환하는 Alledge에서는 name값이 없어 AllNode를 통해 Dict를 만듦
        try:
            objNameListSynset.append(synsDict[str(id)])
            objIdList.append(id)
        except:
            continue
    relabelDict = {objName: i for i, objName in enumerate(objNameListSynset)}  # relabel을 위한 objId
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

    # idNameDict = {ObjIdx : SynsetName} 를 만든기 위해 전체 objIdList를 사용

    attrNameList = []  # name attr  추가를 위한 코드
    for kId in objIdList:
        attrNameList.append(synsDict[str(kId)])
    idNameDict = {str(idx): synsetName for idx, synsetName in zip(synsDict.keys(), attrNameList)}  # name attr  추가를 위한 코드

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
            # df_name = df_name.drop(index=idx)

    # synset에 없어 ''로 append 된 경우 dropna로 해당 행 삭제(해당 relationship 삭제)
    df_edge = df_edge.replace(r'', np.nan, regex=True)
    df_edge = df_edge.dropna()
    # df_edge['objId'] = df_edge['objId'].astype(int)
    # df_edge['subjId'] = df_edge['subjId'].astype(int)
    # df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })

    gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')


    # for index, row in df_edge.iterrows():
    #     gI.nodes[row['objId']]['name'] = row['newObjName']
    #     gI.nodes[row['subjId']]['name'] = row['newSubjName']

    # --------- ^^^ Graph 생성, graph에 name, Origin ObjId를 attribute로 추가함 ^^^ ------------------
    # 자기 자신 참조하는 중복 제거, synset name 적용
    # todo drop na 없어도 잘 동작하는지. 지금 모든 name을 반영하므로, dropna로 삭제되는 행이 없어야 함

    # ----------- vvv 이웃 노드에 동일한 이름을 가진 노드가 5개 이상인 경우 동일 id로 변환 vvv -------------

    nodesList = sorted(list(gI.nodes))
    objIdSet = df_edge['objId'].tolist() + df_edge['subjId'].tolist()
    objNameList = df_edge['newObjName'].tolist() + df_edge['newSubjName'].tolist()

    #위에서 제거된 값을 위해 변경
    objId = df_edge['objId'].tolist()



    subjId = df_edge['subjId'].tolist()
    # totalEmbDict = {Name : Embedding}
    # embDict = { 해당 이미지 내 objId : textEmbedding 값}
    embList = [totalEmbDict[objNameList[i]] for i in range(len(objIdSet))]
    embDict = {idx: emb for idx, emb in zip(objIdSet, embList)}

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
        for i in range(len(sameNums)):
            if sameNums[i] >= 5:
                sameUpp5Idx.append(i)

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
            if len(delTargetIds) != 0:  # todo 해당 라인 삭제 후 돌려보기. 로직상 지워도 오류 X
                delTargetIds = sorted(delTargetIds)
                fId.append(delTargetIds[0])
                totalList.append(delTargetIds)

    # todo 앞서 변경된 Id가 나중에 변경된 Id 값에 의해 왕창 늘어날 가능성 고려 및 코드 수정 필요
    # replaceDict = {delTargetIds : delTargetIds 중 변경 대상이 되는 Id}
    replaceDict = {}
    for i in range(len(totalList)):
        for j in range(len(totalList[i])):
            replaceDict[str(totalList[i][j])] = fId[i]

    newObjList = []
    newSubjList = []
    if (len(replaceDict) != 0):
        for i in objId:
            try:
                newObjList.append(replaceDict[str(i)])  # 교체 대상인 경우 교체
            except KeyError:
                newObjList.append(str(i))  # 이웃 노드에서 중복이 아니어서 교체 대상이 아닌 경우 기존 Id

        for i in subjId:
            try:
                newSubjList.append(replaceDict[str(i)])  # 교체 대상인 경우 교체
            except KeyError:
                newSubjList.append(str(i))  # 이웃 노드에서 중복이 아니어서 교체 대상이 아닌 경우 기존 Id

    if len(newObjList) != 0 :
        newObjId, newSubjId = newObjList, newSubjList
    else :
        newObjId, newSubjId = objId, subjId
    # id 값으로 name 호출 - obj / subj -> idtoName
    newObjName = [idNameDict[str(i)] for i in newObjId]
    newSubjName = [idNameDict[str(i)] for i in newSubjId]
    # #id 값으로 embedding 값 호출 - obj / subj -> Em
    # newObjEmb = [embDict[i] for i in newObjId]
    # newSubjEmb = [embDict[i] for i in newSubjId]

    df_new = pd.DataFrame({"objId": newObjId, "subjId": newSubjId,
                           "objName": newObjName, "subjName": newSubjName})
                         # "objEmb": newObjEmb, "subjEmb": newSubjEmb, })

    df_new['objId'] = df_new['objId'].astype(int)
    df_new['subjId'] = df_new['subjId'].astype(int)

    gI = nx.from_pandas_edgelist(df_new, source='objId', target='subjId')
    for index, row in df_new.iterrows():
        gI.nodes[row['objId']]['name'] = row["objName"]  # name attr
        gI.nodes[row['subjId']]['name'] = row['subjName']  # name attr

        gI.nodes[row['objId']]['originId'] = row['objId']  # originId attr
        gI.nodes[row['subjId']]['originId'] = row['subjId']  # originId attr

    nodesList = sorted(list(gI.nodes))

    for i in range(len(nodesList)):  # nodeId
        nodeId = nodesList[i]
        emb = embDict[nodeId]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
        for j in range(3):  # Embedding 값은 [3,]인데, 원소 각각을 특징으로 node에 할당
            nx.set_node_attributes(gI, {nodeId: emb[j]}, "f" + str(j))

    # node relabel - graph에서 노드 id 0부터 시작하도록 ---
    listA = list(set(newObjId + newSubjId))
    listIdx = range(len(listA))
    dictIdx = {name: value for name, value in zip(listA, listIdx)}
    gI = nx.relabel_nodes(gI, dictIdx)
    gList.append(gI)


with open("./data/networkx_sifted.pickle", "wb") as fw:  # < node[nId]['attr'] = array(float)
    pickle.dump(gList, fw)

with open("./data/networkx_sifted.pickle", "rb") as fr:
    data = pickle.load(fr)

gId = 3
gI = gList[gId]
# print(data)
print('data[gId] : ', gList[gId])
print('data[gId].node : ', gList[gId].nodes(data=True))

print(list(synsDict.values())[0])
print('synsDict len : ', len(list(set(synsDict.values()))))

plt.figure(figsize=[15, 7])
nx.draw(gI, with_labels=True)
plt.show()
