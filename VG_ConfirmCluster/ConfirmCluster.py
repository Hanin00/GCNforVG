import pandas as pd
import sys
sys.path.append(r'./api/visual_genome_python_driver')
from visual_genome import api as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
from collections import Counter
from typing import List

testFile = open('data/cluster10000.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObjList = (readFile.replace("'", '')).split(',')
freObjList = list(map(int, freObjList))

df = pd.DataFrame(freObjList,columns = ['cluster'])

#print(df)

cluster12 = df[df['cluster']== ]
print(cluster12.head(5))
aList = list(cluster12.head(5).index.tolist())

pList = []

for image_id in aList:
    regions = vg.get_region_descriptions_of_image(id=image_id+1)
    for j in range(len(regions)):
        p = regions[j].phrase
        if 'The ' in regions[j].phrase :
            p = p.replace('The ', '')
        if 'the ' in regions[j].phrase :
            p = p.replace('the ', '')
        if 'A ' in regions[j].phrase :
            p = p.replace('A ', '')
        if 'An ' in regions[j].phrase :
            p = p.replace('An ', '')
        if 'a ' in regions[j].phrase :
            p = p.replace('a ', '')
        if 'an ' in regions[j].phrase :
            p = p.replace('an ', '')

        pList.append(p.split())

pList = sum(pList, [])
pList = Counter(pList)
frqHundred = pList.most_common(100)
print(frqHundred)