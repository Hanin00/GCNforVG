# 일단 이미지 하나에 대한 adj(objectidxsubjectid) - nodes, edges / feature matrix(text embedding 값)

import numpy as np
import pandas as pd
import json

import torch
from openpyxl import Workbook
from gensim.models import FastText
from tqdm import tqdm
from collections import Counter


# node : object id 모두 다 합쳐서...
# edge : relationship으로 엮인 관계

# todo vg 드라이버 사용하도록 변경하기
''' Image에 해당하는 모든 ObjectId - ObjectName '''
# Object.json
def AllNodes(data, imgId):
    objectIds = []
    objectNames = []
    objects = data[imgId]["objects"]
    for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
        objectIds.append(objects[j]['object_id'])
        objectNames.append(objects[j]['names'][0])

    return objectIds, objectNames


# ObjId, SubjId, 연결 여부.. 이거 directed로도 만들 수 있을 것 같은데..
# Type 무시 가능한 옵션 있는지 확인하기 -> directed도 가능하지 않나..?
# weight는 일단 다 1
# ObjId, SubjId 각 리스트에 담고, realationship도 혹시 모르니까 일단 만들자...
# SceneGraph.json 사용
''' Relationship로 ObjId-SubjId 간 Edge 추출'''
def AllEdges(data, ImgId):
    objId = []
    subjId = []
    relatiohship = []  # 혹시 모르니까 추가할래 Id 말고 Predicate로~
    edgeId = []  # csv에 저장할 때 한 번에 얹으려고.. 0번부터 시작
    weight = []  # default 1
    # relationships에 따라 중복 허용하고 list에 append하므로 df로 출력하면 엣지를 확인 할 수 있음
    # 이를 이용해 Adj나 Featurematrix의 row 순서를 맞추려면 중복 제거 / 중복 제거 후 순서맞춰서 embedding 값 추가해야함

    imageDescriptions = data[ImgId]["relationships"]

    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        objId.append(imageDescriptions[j]['object_id'])
        subjId.append(imageDescriptions[j]['subject_id'])
        relatiohship.append(imageDescriptions[j]['predicate'])
        edgeId.append(j)
        weight.append(1)

    return objId, subjId, relatiohship, edgeId, weight


# Object Embedding 값 - objectName에 맞춰서 Embedding한 값을 ObjectId에 맞춰(len(objId),10)의 형태로 반환
''' ObjName에 기반한 FastTextEmbedding 값 추출 및 dict(ObjName : Embedding) 반환이었으나 변경
    근데 id:Embedding으로 찾을 수 있어야 함.
    feature의 type은 torch.float 형 <- dgl.from_networkx를 하려면 attr이 tensor로 변경 가능해야함
    -> torch.FloatTensor 형이 아닌 단순 array(float) 형으로 변경함 <networkx1000_noTensor.pickle임
    ObjName에 기반한 FastTextEmbedding 값 추출 및 dict(ObjId : Embedding) 반환 '''
def FeatEmbeddPerImg(objectIds, objectNames):
    # a = []
    # a.append(objectNames)
    # model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)
    model = FastText()
    #vocab가 너무 적은 경우(5개 정도로) 정상 작동하지 않아 id 242의 objectName을 임의로 삽입함 -> Top Object를 넣는 게 더 나을 것 같기두..
    if(len(objectNames) <= 20):
        objectNames += ['road', 'taxi', 'door', 'taxi', 'wagon', 'man', 'stairs', 'suit', 'street', 'van', 'sign', 'car', 'symbol', 'car', 'trafficcone', 'ground', 'tag', 'lights', 'woman', 'cars', 'people','car', 'people', 'steps', ]
    model.build_vocab(objectNames)
    model = FastText(objectNames, vector_size=3, workers=4, sg=1, word_ngrams=1)
    #model.build_vocab(objectNames)
    embedding = []
    for i in objectNames: #objName 개수만큼만 반복하니까 vocab에 추가해 준 거 신경 X. Id:Embedding 값으로 dict 생성
        embedding.append(model.wv[i])
    # objectNames, Embedding 형태로 Dict 저장
    embDict = {name: value for name, value in zip(objectIds, embedding)}
    #embDict = {name: torch.FloatTensor(value) for name, value in zip(objectIds, embedding)}

    return embDict

'''
    전체 word에 대한 fasttext Embedding값  - 01
'''
def FeatEmbeddPerTotal(objectNames):
    # a = []
    # a.append(objectNames)
    # model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)
    model = FastText()
    #vocab가 너무 적은 경우(5개 정도로) 정상 작동하지 않아 id 242의 objectName을 임의로 삽입함 -> Top Object를 넣는 게 더 나을 것 같기두..
    model.build_vocab(objectNames)
    model = FastText(objectNames, vector_size=3, workers=4, sg=1, word_ngrams=1)
    #model.build_vocab(objectNames)
    embedding = []
    for i in objectNames: #objName 개수만큼만 반복하니까 vocab에 추가해 준 거 신경 X. Id:Embedding 값으로 dict 생성
        embedding.append(model.wv[i])
    # objectNames, Embedding 형태로 Dict 저장
    totalEmbDict = {name: value for name, value in zip(objectNames, embedding)}
    #embDict = {name: torch.FloatTensor(value) for name, value in zip(objectIds, embedding)}

    return totalEmbDict

'''
    전체 word에 대한 fasttext Embedding값  - 02 < 각 이미지 별 node Id/Name 매칭 
    Dict(wordEmbedding - Word:Embedding) 을 이용해 Id : Embedding Dict를 반환하도록 변경    
'''

def MatchDictImage(objectIds, objectNames, totalEmbDict):
    embList = []
    for i in objectNames: #objName 개수만큼만 반복하니까 vocab에 추가해 준 거 신경 X. Id:Embedding 값으로 dict 생성
        embList.append(totalEmbDict[i])
    # objectNames, Embedding 형태로 Dict 저장
    embDict = {name: value for name, value in zip(objectIds, embList)}

    return embDict



''' objId에 맞는 FeatureEmbedding값으로 matrix 만듦 '''
def FeatureMatrix(objIdList, objDict, embDict):
    featM = []
    for i in range(len(objIdList)):
        featM.append(embDict[objDict[objIdList[i]]])

    return np.array(featM)


import json
import pickle
import time
import networkx as nx
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm


# scene_graph.json -> adjMatrix/FeatureMatrix 만들 때 오류나서 제외한 사진들이 있음
# 해당 이미지를 제외하고 label 값을 추출해야해서 따로 처리함
def siftCluster():
    testFile = open('./data/cluster5000.txt', 'r')
    readFile = testFile.readline()
    oldLabel = (readFile[0:-1].replace("'", '')).split(',')
    label = []

    siftListId = [50, 60, 156, 241, 284, 299, 317, 371, 403, 432, 512, 520, 647, 677, 745, 867, 930, 931, 1102, 1116,
                  1136, 1174, 1212, 1239, 1256, 1278]

    for i in range(len(oldLabel)):
        if i in siftListId:
            continue
        else:
            label.append(oldLabel[i])
        if (len(label) == 1000):
            break

    print(len(label))
    with open("./data/clusterSifted1000.pickle", "wb") as fw:
        pickle.dump(label, fw)


''''
    data
    data.y : cluster 값, list 형태로
    data.y[mask] : cluster label 이라는데 일단 data.y[mask] = data.y?
    data.val_mask : random seed 정해서 비율 idx 범위정하기?
    data.train_mask :
    data.test_mask :    

    data.x : feature matrix
    data.edge_list : sparse adjacency list
    data.batch : batch vector... ????? 그래프 노드 개수??? 그래프??? 그래프들? vector라는데 networkx 객체가 아닌건가?? Long tensor라는데 이게 뭐지.. < optional 이라는데

    data.num_nodes
    data.num_classes : 클래스 개수, 10
    data.num_graphs : 그래프 개수, 1000 

    data.num_node_features  << 없을 수 있음    

    data.test_neg_edge_index / test_pos_edge_index    
    data.train_neg_edge_index / train_pos_edge_index    
'''


def mkEdgelistPer1(netx):
    a = netx.edges.data()
    sr = pd.DataFrame(a)
    sr.columns = ['n1', 'n2', 'feature']
    n1 = sr['n1'].values.tolist()
    n2 = sr['n2'].values.tolist()
    return n1, n2


# 일단 local로 가지고 있으면서 모델 돌리기
def mkEdgelistPerRange(netx, num):
    n1 = []
    n2 = []
    for idx in tqdm(range(num)):
        a = netx[idx].edges.data()
        sr = pd.DataFrame(a)

        sr.columns = ['n1', 'n2', 'feature']
        # sr.columns = ['n1', 'n2']
        n1.append(sr['n1'].values.tolist())
        n2.append(sr['n2'].values.tolist())
    return n1, n2


# '''
#     edgeList local 저장 및 확인
# '''
# with open("./data/_networkx1000.pickle", "rb") as fr:
#     netx = pickle.load(fr)
#
# n1, n2 = mkEdgelistPerRange(netx, 1000)
# edgeList = [n1, n2]
#
# with open('./data/edgeList1000.pickle', 'wb') as f:
#     pickle.dump(edgeList[:1001], f)
#
# with open("./data/edgeList1000.pickle", "rb") as fr:
#     edgeList = pickle.load(fr)
#
# print(len(edgeList))
# print(edgeList[0])
# print(len(edgeList[0][0]))
# print(len(edgeList[0][1]))
