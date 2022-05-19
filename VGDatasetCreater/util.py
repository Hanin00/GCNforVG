# 일단 이미지 하나에 대한 adj(objectidxsubjectid) - nodes, edges / feature matrix(text embedding 값)

import numpy as np
import pandas as pd
import json
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
''' ObjName에 기반한 FastTextEmbedding 값 추출 및 dict(ObjId : ObjName) 반환 '''
def FeatEmbeddPerImg(objectNames):
    # a = []
    # a.append(objectNames)
    # model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)
    model = FastText()
    #vocab가 너무 적은 경우(5개 정도로) 정상 작동하지 않아 id 242의 objectName을 임의로 삽입함 -> Top Object를 넣는 게 더 나을 것 같기두..
    if(len(objectNames) <= 20):
        objectNames += ['road', 'taxi', 'door', 'taxi', 'wagon', 'man', 'stairs', 'suit', 'street', 'van', 'sign', 'car', 'symbol', 'car', 'trafficcone', 'ground', 'tag', 'lights', 'woman', 'cars', 'people','car', 'people', 'steps', ]
    model.build_vocab(objectNames)
    model = FastText(objectNames, vector_size=10, workers=4, sg=1, word_ngrams=1)
    #model.build_vocab(objectNames)
    embedding = []
    for i in objectNames:
        embedding.append(model.wv[i])
    # objectNames, Embedding 형태로 Dict 저장
    embDict = {name: value for name, value in zip(objectNames, embedding)}

    return embDict


''' objId에 맞는 FeatureEmbedding값으로 matrix 만듦 '''
def FeatureMatrix(objIdList, objDict, embDict):
    featM = []
    for i in range(len(objIdList)):
        featM.append(embDict[objDict[objIdList[i]]])

    return np.array(featM)

