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
def extractNoun(noun, synsDict, synNameCnter) :
    conllDict = {word: tag for word, tag in conll2000.tagged_words(tagset='universal')}

    words = noun.split(' ')
    # noun 판별
    nouns = []
    for i in words :
        try:
            if conllDict[i] == 'NOUN':
                nouns.append(i)
        except :
            continue
    # synset에 해당되는 noun이 있는지 판별
    nInSynsDictList = []
    if len(nouns) != 0 :
        for i in nouns :
            try :
                nInSynsDictList.append(synsDict[i])
            except :
                continue

    # synset에 해당되는 noun 중 언급이 많은 단어 선택
    name = ''
    cnt = 0
    if len(nInSynsDictList) != 0 :
        for i in nInSynsDictList:
            if cnt < synNameCnter[i] :
                name = i
                cnt = synNameCnter[i]
    else :
        name = "_".join(sorted(nouns))
    return name


gList = []
imgCnt = 10
start = time.time()
with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
end = time.time()
print(f"파일 읽는데 걸리는 시간 : {end - start:.5f} sec")  # 파일 읽는데 걸리는 시간 : 24.51298 sec


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

for i in range(len(nonSysnNameList)) :
    try :
        sameNameId = get_key(originDict, nonSysnNameList[i]) # 1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우,
        synsDict[nonSysnIdList[i]] = synsDict[sameNameId]

    except :
            #todo 2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
        name = extractNoun(nonSysnNameList[i], synsDict, synNameCnter)
        synsDict[nonSysnIdList[i]] = name


# #위에서 만든 synset Dict를 이용해 totalEmbedding 값을 만듦(fasttext)
# objNamesList = []
# newObjIdListforSynset = []
# for imgId in tqdm(range(imgCnt)):
#     objectIds, objectNames = ut.AllNodes(data, imgId)
#     for id in objectIds:
#         try:
#             objNamesList.append(synsDict[str(id)])  # synset Id로 변경
#             newObjIdListforSynset.append(id)  # synset 이 없어 Dict에 없는 ObjId는 제외함
#         except:
#             continue


objectNameList = list(set(list(synsDict.values())))
model, totalEmbDict = ut.FeatEmbeddPerTotal_model(objectNameList)
with open("./data/totalEmbDict.pickle", "wb") as fw:
    pickle.dump(model, fw)
# --------------------------- ^^^ synset Dict, Total Embedding(fasttext 값)^^^ ---------------------------

for i in tqdm(range(imgCnt)):
    objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data, i)
    # networkX graph 객체 생성 ---
    # 이미지 내의
    objIdSet, objNameList = ut.AllNodes(data, i)

    objIdList = []
    objNameListSynset = []
    for id in objIdSet :  #relationship으로 부터 반환하는 Alledge에서는 name값이 없어 AllNode를 통해 Dict를 만듦
        try :
            objNameListSynset.append(synsDict[str(id)])
            objIdList.append(id)
        except :
            continue
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
    relabelDict = {objName: i for i, objName in enumerate(objNameListSynset)}  #relabel을 위한 objId
    originIdList = []
    attrNameList = []    #name attr  추가를 위한 코드
    newObjIdList = []
    for kId in objIdList:
        originIdList.append(str(kId))
        attrNameList.append(synsDict[str(kId)])
        newObjIdList.append(relabelDict[synsDict[str(kId)]])        # name attr  추가를 위한 코드

    idIdxDict = {name: value for name, value in zip(originIdList, newObjIdList)}
    idNameDict = {name: value for name, value in zip(originIdList, attrNameList)}    #name attr  추가를 위한 코드
    idxOriginIdDict = {str(name): value for name, value in zip(newObjIdList, originIdList)}   #origin attr 추가를 위한 코드

    newObjId = []
    newSubjId = []

    newObjName = []       #name attr  추가를 위한 코드
    newSubjName = []      #name attr  추가를 위한 코드

    originObjIdList = []
    originSubjIdList = []

    #자기 자신에 참조하는 node 삭제
    for oId in objIdList:
        print('oId : ', oId)
        try :
            newObjId.append(idIdxDict[oId])
            newObjName.append(idNameDict[oId])
            originObjIdList.append(idxOriginIdDict[idIdxDict[oId]])
            print('1 : ', idIdxDict[oId])
            print('1 : ',idNameDict[oId])
            print('1 : ',idxOriginIdDict[idIdxDict[oId]])

            sys.exit()

        except :
            newObjId.append('')
            newObjName.append('')
            originObjIdList.append('')
            print('안됨')


    for sId in subjId:
        try:
            newSubjId.append(idIdxDict[sId])
            newSubjName.append(idNameDict[sId])
            originSubjIdList.append(idxOriginIdDict[idIdxDict[sId]])
        except :
            newSubjId.append('')
            newSubjName.append('')
            originSubjIdList.append('')



    recurRowId = []
    for j in range(len(newObjId)):
        if newObjId[j] == newSubjId[j]:
            recurRowId.append(j)

    print(len(newObjId))
    print(len(newObjName))
    print(len(originObjIdList))

    sys.exit()
    df_edge = pd.DataFrame({"objId": newObjId, "subjId": newSubjId,"newObjName": newObjName, "newSubjName": newSubjName,'originObjId' :originObjIdList ,'originSubjId' : originSubjIdList})

    # todo 자기 자신을 참조하는 노드의 relationship 삭제
    if recurRowId != 0:
        for idx in recurRowId:
            df_edge = df_edge.drop(index=idx)
            #df_name = df_name.drop(index=idx)

    # todo synset에 없어 ''로 append 된 경우 dropna로 해당 행 삭제(해당 relationship 삭제)
    #df_edge = df_edge.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
    df_edge = df_edge.replace(r'', np.nan, regex=True)
    df_edge = df_edge.dropna()

    df_edge['objId'] = df_edge['objId'].astype(int)
    df_edge['subjId'] = df_edge['subjId'].astype(int)

   # df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })
    gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')

    for index, row in df_edge.iterrows():
        gI.nodes[row['objId']]['name'] = row['newObjName']
        gI.nodes[row['objId']]['objId'] = row['originObjId']
        gI.nodes[row['subjId']]['name'] = row['newSubjName']
        gI.nodes[row['subjId']]['objId'] = row['originSubjId']

    print("gI.nodes : ", gI.nodes)
    nodesList = sorted(list(gI.nodes))
    print("nodesList : ", nodesList)
    print("len(nodesList) : ", len(nodesList))

    #embDict = ut.MatchDictImage(newObjId+newSubjId,  newObjName+newSubjName, totalEmbDict)
    # node attribute 부여 ---
    objIdSet = df_edge['objId'].tolist() + df_edge['subjId'].tolist()
    objNameList =df_edge['newObjName'].tolist() + df_edge['newSubjName'].tolist()  #todo
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
        print(nodeId)
        sys.exit()
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

    replaceDict = {}
    for i in range(len(totalList)):
        for j in range(len(totalList[i])):
            replaceDict[str(totalList[i][j])] = fId[i]

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

        newObjId, newSubjId =  newObjList, newSubjList
        df_new = pd.DataFrame({"objId": newObjId, "subjId": newSubjId, })
        print('df_new')
        print(df_new)
        gI = nx.from_pandas_edgelist(df_new, source='objId', target='subjId')
        nodesList = sorted(list(gI.nodes))


    for i in range(len(nodesList)):  # nodeId
        nodeId = nodesList[i]
        emb = embDict[nodeId]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
        print(nodeId)
        for j in range(3):  # Embedding 값은 [3,]인데, 각 원소를 특
            # 징으로 node에 할당
            nx.set_node_attributes(gI, {nodeId: emb[j]}, "f" + str(j))

    print(gI.nodes(data="name"))
    print(gI.nodes(data="objId"))
    sys.exit()


    # graph에서 노드 id 0부터 시작하도록 ---
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
#print(data)
print('data[gId] : ', gList[gId])
print('data[gId].node : ', gList[gId].nodes(data=True))

print(list(synsDict.values())[0])
print('synsDict len : ',len(list(set(synsDict.values()))))

plt.figure(figsize=[15, 7])
nx.draw(gI, with_labels=True)
plt.show()