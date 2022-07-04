import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image
import networkx as nx
import requests
import json
import pickle
import sys
import time
import numpy
import pandas as pd

with open('./data/scene_graphs.json') as file:  # open json file
     data = json.load(file)



gIdList = []
objIdList = []
synsList = []

ngIdList = []
nobjIdList = []
nsynsList = []



for gId in range(1000) :
    imageDescriptions = data[gId]["objects"]
    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            try :
                names = imageDescriptions[j]['names'][0]
                syns = imageDescriptions[j]['synsets'][0].splits('.')[0]
                id = imageDescriptions[j]['object_id']
                if names == "top":
                    print(names)
                    print(gId)
                    print(syns)
                    gIdList.append(gId)
                    objIdList.append(j)
                    synsList.append(syns)
                    if syns != "blouse" :
                        ngIdList.append(gId)
                        nobjIdList.append(j)
                        nsynsList.append(syns)
            except :
                continue

df1 = pd.DataFrame({"gIdList" : gIdList, "objIdList" : objIdList,"synsList" : synsList} )
df2 = pd.DataFrame({"ngIdList" : ngIdList, "nobjIdList" : nobjIdList,"nsynsList" : nsynsList} )

print("df1")
print(df1)

print("df2")
print(df2)

sys.exit()

# with open("data/networkx_ver1.pickle", "rb") as fr:
#     data = pickle.load(fr)
#
# print(data[114].nodes(data='name'))
# print(data[114].nodes(data=True))
# sys.exit()
#
# lista = []
#
# for ImgId in range(1000) :
#     imageDescriptions = data[ImgId]["objects"]
#     for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
#         names = imageDescriptions[j]['names'][0]
#         if names == 'top' :
#             try :
#                 syns = imageDescriptions[j]['synsets'][0].split('.')[0]
#                 lista.append(syns)
#                 if syns == 'blouse':
#                     print('gid : ', ImgId)
#             except :
#                 continue
#
#
# print('blouse' in lista)
# sys.exit()


















print(data[471])
print(data[471].nodes(data='name'))
print(data[471].nodes(data='originId'))


listb = ['blouse', 'tree','window','building', 'person', 'road', 'land', 'sidewalk']
for gId in range(len(data)) :
    lista = [i[1] for i in data[gId].nodes(data = 'name')]
    lista = sorted(lista)
    if lista ==sorted(listb) :
        print(gId)
