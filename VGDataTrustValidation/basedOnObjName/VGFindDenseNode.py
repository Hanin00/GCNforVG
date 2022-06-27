import matplotlib.pyplot as plt
import networkx as nx
import json
import pandas as pd
import pickle
import VGDataTrustValidation.util as ut


def imgShow(nexG):
    nx.draw(nexG, with_labels=True)
    plt.show()


#
# try:
#     from StringIO import StringIO as ReadBytes
# except ImportError:
#     print("Using BytesIO, since we're in Python3")
#     #from io import StringIO #..
#     from io import BytesIO as ReadBytes




'''
    target node : neighbor node 제일 많은 node Id
    1. target node 의 1홉 기준 연결된 노드가 몇 개인지
        - 새로운 objId, subjId, objName, subjName
    2. target node 의 name을 갖는 Object의 위치 object로 표시
    
    <- 이때, text name 의 첫번째만 허용하므로, [table, desk] , [desk], [table]이라는
    object가 3개 있을 경우, 0번과 2번은 동일 node지만 0번과 1번은 다른 node임
    
'''
#
with open("../data/networkx_name.pickle", "rb") as fr:
    data = pickle.load(fr)

gId = 0
tG = data[gId]

print(data[0])
print(data[0].nodes)

def findDenseNode(tG) :

    neighborsNum = []
    nodeList = sorted(tG.nodes)

    for i in nodeList:
        # 일단 append 후 max 값의 idx 찾아서, dense node idx 찾기
        neighborsNum.append(len(list(tG.neighbors(i))))

    print('neighborsNum : ', neighborsNum)
    densenodeIdx = nodeList[neighborsNum.index(max(neighborsNum))]

    print(tG)
    print('densenodeIdx : ', densenodeIdx)
    print('densenodeNeighborsNum : ', len(list(tG.neighbors(densenodeIdx))))
    print('densenodeNeighbors : ', list(tG.neighbors(densenodeIdx)))
    imgShow(tG)

    return densenodeIdx


def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

with open('../data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data, gId)
objIdSet, objNameList = ut.AllNodes(data, gId)   #name Listd의 [0]번만 있음

objNIDict = {id: name for id, name in zip(objIdSet, objNameList)}

objName = []
subjName = []

for oId in objId:
    objName.append(objNIDict[oId])
for sId in subjId:
    subjName.append(objNIDict[sId])

df1 = pd.DataFrame({"objIdSet": objIdSet, "objNameList": objNameList})
df2 = pd.DataFrame({"objId": objId, "subjId": subjId,"objName" : objName,"subjName": subjName,})
print(df1)
print(df2)

# newSub,Obj에 맞는 걸로 Name 알아내서 objId 값 List 만들어서 이미 만들어진 모듈에 넣어서 visualize


relabelDict = {objName: i for i, objName in enumerate(objNameList)}
rvelabelDict = {i: objName for i, objName in enumerate(objNameList)}
#print(get_key(relabelDict, 2)) #원래의 name 값 얻을 수 있음

newObjIdList = []
for kId in range(len(objIdSet)):
    newObjIdList.append(relabelDict[objNameList[kId]])
#그래프에서 nodes id  얻어서 그에 대한 name을 얻고, 해당 name을 가진 objId을 objIdSet에서 추출


densenodeIdx = findDenseNode(tG)
originName = get_key(relabelDict, densenodeIdx)
print('dense node Idx ',densenodeIdx,'의 원래 name : ',originName)

names = []
for i in (list(tG.neighbors(densenodeIdx))):
    print(densenodeIdx,'의 neghibor', i,'의 name : ',rvelabelDict[i])

for i in (list(tG.neighbors(densenodeIdx))):
    print(densenodeIdx,'의 neghibor', i,'의 name : ',get_key(rvelabelDict, i))

sameNameIds = []
for i in range(len(objNameList)) :
    if objNameList[i] == originName :
        sameNameIds.append(objIdSet[i])

print('sameNameIds : ', sameNameIds)

#neighborHood








# 단일 node에 이웃 노드 이름이 5개 이상 겹치는 경우, 이름이 같은 오브젝트들을 모두 하나의 node로 묶고 오브젝트를 삭제함
# neighbor node의 범위가 중복되는 경우 id를 name 기준으로 합친 후 새로운 id를 부여하고 근처에 위치하는 동일 이름을 가진 obj의 id값이ㅡㄹ
# 새로 부여한 id로 치환