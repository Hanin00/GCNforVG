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

'''
    Subgraph의 node attribute 중 origin Id를 통해 
    Subgraph와 대상 그래프 간 Object Name값이 일치하는 Node를 
    Img(대상 그래프)에서 Bounding Box를 통해 확인
'''


try:
    from StringIO import StringIO as ReadBytes
except ImportError:
    print("Using BytesIO, since we're in Python3")
    #from io import StringIO #..
    from io import BytesIO as ReadBytes



with open('./data/scene_graphs.json') as file:  # open json file
    sceneJson = json.load(file)
with open("./data/networkx_sifted_20.pickle", "rb") as fr:
    graphs = pickle.load(fr)


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
    objectList = [IdCoNameDict[idx] for idx in nodes]

    return objectList



#visualize with bounding box
def visualize_regions(image, objectsList):
    response = requests.get(image.url)
    img = PIL_Image.open(ReadBytes(response.content))
    plt.imshow(img)
    ax = plt.gca()
    for object in objectsList:
        ax.add_patch(Rectangle((object.x, object.y),
                               object.w, object.h,
                               fill=False,
                               edgecolor='red', linewidth=3))
        ax.text(object.x, object.y, object.phrase, style='italic',
                bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    #plt.tick_params(labelbottom='off', labelleft='off')
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.show()


#data load
gId = 0
image = vg.get_image_data(gId+1)
image = vg.get_image_data(image_id)
G = graphs[gId]
print(G)
print(G.nodes(data=True))

#visualize on plt
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
objectList = makeObjectsInSubG(gId, G)
print(objectList)
visualize_regions(image, objectList)