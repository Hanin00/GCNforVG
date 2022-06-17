from visual_genome import api as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
import json
import sys
import time
import pandas as pd
from collections import Counter

import pickle

#imageId는 1부터 시작함 / 편의를 위해 imgId로 표기하고, 실제 img id가 필요한 곳에서 +1 함
# image 가 5면 4 적으면 됨~_~ list는 0부터 시작하니까~_~
imgId = 5

#data load
with open('./data/scene_graphs.json') as file:  # open json file
    sceneJson = json.load(file)




'''
    대조군  
    (비교적) 정상 그래프 :  3,4,491,545, 636?
    밀집 그래프 :  5,289,492,544,637


    대조군의 
    1. relationship 내의 obj-subj Id df로 보기 편하게 출력
    2-1. 해당 이미지의 Scene graph에서 object Id, Object Name, x,y,width, height 찾아서 바운딩 박스로 표현
    2-2. 빈출 object Id(obj, subj 둘 다) 찾아서 해당 object 표시 
        a. df 에서 해당 object Id를 포함하는 row 표시하고, graph에서 연결된 edge 개수와 동일한지 확인
        b. 빈출 object Id와 relationship으로 묶인 subject Id를 bounding box로 표시해 확인
'''
# 1. relationship 내의 obj-subj Id df로 보기 편하게 출력
# 0번 scenegraph의
relationships = sceneJson[imgId]['relationships']
print('relationships : ', len(relationships))

print("sceneJson image_id : ", sceneJson[imgId]['image_id'])

objSuj = []
for rel in range(len(relationships)) :
    objSuj.append((relationships[rel]['relationship_id'], relationships[rel]['object_id'], relationships[rel]['subject_id']))


df = pd.DataFrame(list(objSuj))
df.columns = ['relId', 'objId','subId']

print('total')
print(df)

print('used objectId count(non-overlap) : ', len(list(set(df['objId'].to_list() +df['subId'].to_list()))))

# 2-1. 해당 이미지의 Scene graph에서 object Id, Object Name, x,y,width, height 찾아서 바운딩 박스로 표현
try:
    from StringIO import StringIO as ReadBytes
except ImportError:
    print("Using BytesIO, since we're in Python3")
    #from io import StringIO #..
    from io import BytesIO as ReadBytes

def visualize_regions(image, objects, color, freqId = -1):
    response = requests.get(image.url)
    img = PIL_Image.open(ReadBytes(response.content))
    plt.imshow(img)
    ax = plt.gca()
    for obj in objects:
        if obj['object_id'] == freqId :
            ax.add_patch(Rectangle((obj['x'], obj['y']),
                                   obj['w'], obj['h'],
                                   fill=False,
                                   edgecolor='red', linewidth=3))
            # edgecolor='red', linewidth=3))
            ax.text(obj['x'], obj['y'], obj['names'], style='normal',
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
            print("freqObjId : ", freqId)
            print("freqObjName : ", obj['names'])

        else :
            ax.add_patch(Rectangle((obj['x'], obj['y']),
                                   obj['w'], obj['h'],
                                   fill=False,
                                   edgecolor=color, linewidth=3))
            # edgecolor='red', linewidth=3))
            ax.text(obj['x'], obj['y'], obj['names'], style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 10})



    fig = plt.gcf()
    #plt.tick_params(labelbottom='off', labelleft='off')
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.show()

#visualize on plt
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

image_id = sceneJson[imgId]['image_id']
image = vg.get_image_data(image_id)
objects = sceneJson[imgId]['objects']
print('objects : ', objects)
image_id = sceneJson[imgId]['image_id']
print('object num : ', len(objects))
visualize_regions(image, objects[:], 'green')


#   2-2. 빈출 object Id(obj, subj 둘 다) 찾아서 해당 object 표시
import itertools
objIds = list(itertools.chain(*list(objSuj)))
objIds = Counter(objIds)


freq = objIds.most_common(1)
freq = list(freq)
freqId = freq[0][0]
freqCnt = freq[0][1]
#print('frequencyId : ',freqId)

#  2-2 a. df 에서 해당 object Id를 포함하는 row 표시하고, graph에서 연결된 edge 개수와 동일한지 확인
condition = (df.objId == freqId) | (df.subId == freqId)
df2 = df[condition] # == df.loc[condition]
df2.index.names = ['idx']
df2 = df2.reset_index()
#df2['idx'] = df2['idx']


#todo 동일 objId-subId 갖는 relationship Id 의 name 값 확인
df2RelIdName = {}
df2RelName = []
df2RelId = df2['relId'].to_list()

for rel in range(len(relationships)) :
    df2RelIdName[relationships[rel]['relationship_id']] = relationships[rel]['predicate']

for relId in df2RelId :
    df2RelName.append(df2RelIdName[relId])
df2['predicate'] = df2RelName


#print(df2['idx'])

# 2-2. 빈출 object Id(obj, subj 둘 다) 찾아서 해당 object 표시
df2Idx = df2['idx'].to_list()
df2ObjId = df2['objId'].to_list()
df2SubId = df2['subId'].to_list()
freObjIds = list(set(df2ObjId+df2SubId))

freObjectsList = []
for objId in freObjIds :
    for i in range(len(objects)) :
        if objId == objects[i]['object_id'] :
            freObjectsList.append({'object_id' : objects[i]['object_id'],
                                   'x' : objects[i]['x'],
                                   'y' : objects[i]['y'],
                                   'w' : objects[i]['w'],
                                   'h' : objects[i]['h'],
                                   'names' : objects[i]['names'],
                                   })


df2objId2Name = []
df2subjId2Name = []
for objId in df2ObjId :
    df2objId2Name.append(next((item for item in freObjectsList if item['object_id'] == objId),False)['names'])

for subjId in df2SubId :
    df2subjId2Name.append(next((item for item in freObjectsList if item['object_id'] == subjId),False)['names'])

df2['objName'] = df2objId2Name
df2['subjName'] = df2subjId2Name





print(df2)

visualize_regions(image, freObjectsList[:], 'blue', freqId)


#
# with open("./data/image1.pickle", "wb") as fw:
#     pickle.dump(, fw)
#
# with open("./data/image1.pickle", "rb") as fr:
#     data = pickle.load(fr)