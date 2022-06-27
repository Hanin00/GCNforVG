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
from visual_genome import api as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
import json
import pickle
import sys
import time
import numpy


try:
    from StringIO import StringIO as ReadBytes
except ImportError:
    print("Using BytesIO, since we're in Python3")
    #from io import StringIO #..
    from io import BytesIO as ReadBytes



#relationship 기반의 obj-subj Graph visualize를 Image에 boundign box로 표현

with open('./data/scene_graphs.json') as file:  # open json file
    sceneJson = json.load(file)

'''
    그래프 내 모든 노드의 이미지 내 위치값과 이름을 갖는 Dict List 반환
'''
def makeObjectsInSubG(gId, G) :
    # 0번 scenegraph의
    objects = sceneJson[gId]['objects']

    # ObjId : coordinate(x,y,w,h, name)
    IdCoNameDict = {}

    for i in range(len(objects)):
        object_id = objects[i]['object_id']
        obj = objects[i]
        IdCoNameDict[object_id] = {'x': obj['x'], 'y': obj['y'], 'w': obj['w'], 'h': obj['h'], 'name': obj['names']}

    nodes = G.nodes
    objectList = []
    for idx in nodes :
        objectList+= [IdCoNameDict[idx]]

    return objectList


#visualize with bounding box(use Visual Genome api)
def patchOnImgApi(image, objectsList, denseNode):
    response = requests.get(image.url)
    img = PIL_Image.open(ReadBytes(response.content))
    plt.imshow(img)
    ax = plt.gca()
    if len(denseNode) != 0 :
        for object in denseNode:
            ax.add_patch(Rectangle((object['x'], object['y']),
                                   object['w'], object['h'],
                                   fill=False,
                                   edgecolor='green', linewidth=3))
            ax.text(object['x'], object['y'], object['name'], style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})


    for object in objectsList:
        ax.add_patch(Rectangle((object['x'], object['y']),
                               object['w'], object['h'],
                               fill=False,
                               edgecolor='red', linewidth=3))
        ax.text(object['x'], object['y'], object['name'], style='italic',
                bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    #plt.tick_params(labelbottom='off', labelleft='off')
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.show()

gId = 0
with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)

objectIds, objectNames = ut.AllNodes(data,gId)
objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data,gId)

dictIdx = {name:value for name, value in zip(objectIds, objectNames)}

df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })
G = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')


nodesList = sorted(list(G.nodes))
for idx in nodesList:  # nodeId
    nx.set_node_attributes(G, dictIdx[idx], 'name')

print(G)
print(G.nodes(data=True))

makeObjectsInSubG(gId, G)

#visualize on plt
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
denseNode = []
objectList = makeObjectsInSubG(gId, G)
#denseNode, objectList = mkDenseObjInSubG(gId, G)


patchOnImgApi(image, objectList, denseNode)