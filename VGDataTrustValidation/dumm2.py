# install NLTK Data
# ref) https://www.nltk.org/install.html
from visual_genome import api as vg
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import json
import nltk
from nltk.corpus import conll2000
import sys
import time



with open("./data/networkx_sifted.pickle", "rb") as fr:
    graphs = pickle.load(fr)

with open('./data/scene_graphs.json') as file:  # open json file
    sceneJson = json.load(file)

gId = 0
image = vg.get_image_data(gId+1)
G = graphs[gId]

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

for i in G.nodes :
    if(len(list(G.neighbors(i))) >= 5) :
        denseNode.append(i)
    elif(len(list(G.neighbors(i))) >= dNode) :
        dNode = i

if (dNode in denseNode) != 0 :
    denseNode.append(dNode)

objectList = []

[[objectList.append(IdCoNameDict[objects[idx]['object_id']]) for idx in G.neighbors(nodeIdx)] for nodeIdx in denseNode]

objectList.append(IdCoNameDict[objects[idx]['object_id']] for idx in denseNode)


print(objectList)