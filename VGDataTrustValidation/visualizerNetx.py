from visual_genome import api as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import networkx as nx
import requests
import json
import pickle
import sys
import time
import numpy


'''
    Subgraph의 node attribute 중 origin Id를 통해 
    Subgraph와 대상 그래프 간 Object Name값이 일치하는 Node를 
    Img(대상 그래프)에서 Bounding Box를 통해 확인
'''

with open("./data/networkx_ver1.pickle", "rb") as fr:
    graphs = pickle.load(fr)
# a = numpy.mean([len(graph.nodes) for graph in graphs])
# print(a)

try:
    from StringIO import StringIO as ReadBytes
except ImportError:
    print("Using BytesIO, since we're in Python3")
    #from io import StringIO #..
    from io import BytesIO as ReadBytes


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

    nodes = G.nodes(data="originId")
    nodes = [n[1] for n in nodes]
    objectList = []
    for idx in nodes :
        objectList+= [IdCoNameDict[idx]]

    return objectList


'''
    그래프 내 이웃 노드 수가 가장 많은 노드와 그 이웃 노드의 이미지 내 정보를 갖는 Dict List 반환
'''
def mkDenseObjInSubG(gId, G) :
    # 0번 scenegraph의
    objects = sceneJson[gId]['objects']

    # ObjId : coordinate(x,y,w,h, name)
    IdCoNameDict = {}

    for i in range(len(objects)):
        object_id = objects[i]['object_id']
        obj = objects[i]
        IdCoNameDict[object_id] = {'x': obj['x'], 'y': obj['y'], 'w': obj['w'], 'h': obj['h'], 'name': obj['names']}

    # nodes = G.nodes(data="originId")
    # nodes = [n[1] for n in nodes]
    denseNode = []
    dNode = 0

    for i in G.nodes:
        if (len(list(G.neighbors(i))) >= 5):
            denseNode.append(i)
        elif (len(list(G.neighbors(i))) >= dNode):
            dNode = i

    if (dNode in denseNode) != 0:
        denseNode.append(dNode)
    objectList = []
    [[objectList.append(IdCoNameDict[objects[idx]['object_id']]) for idx in G.neighbors(nodeIdx)] for nodeIdx in
     denseNode]

    denseNodeList = []
    [denseNodeList.append(IdCoNameDict[objects[idx]['object_id']]) for idx in denseNode]



    return denseNodeList,objectList



#visualize with bounding box(use Visual Genome api)
def patchOnImgApi(image, objectsList, denseNode):
    response = requests.get(image.url)
    img = PIL_Image.open(ReadBytes(response.content))
    plt.imshow(img)
    ax = plt.gca()
    if len(denseNode) !=0 :
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

#visualize with bounding box(Local Image File)
def patchOnImgLocal(imagepath, objectsList):

    img = PIL_Image.open(imagepath)
    plt.imshow(img)
    ax = plt.gca()
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



#data load
gId = 0
G = graphs[gId]

image = vg.get_image_data(gId+1)


plt.figure(figsize=[15, 7])
nx.draw(G, with_labels=True)
plt.show()

print(G)
print(G.nodes(data=True))

#visualize on plt
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
denseNode = []
objectList = makeObjectsInSubG(gId, G)
#denseNode, objectList = mkDenseObjInSubG(gId, G)



imagepath = './data/python.png'

patchOnImgApi(image, objectList, denseNode)