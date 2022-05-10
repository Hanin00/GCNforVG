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

testFile = open('./cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObjList = (readFile.replace("'", '')).split(',')
freObjList = list(map(int, freObjList))

df = pd.DataFrame(freObjList,columns = ['cluster'])

#print(df)

cluster12 = df[df['cluster']== 14]
print(cluster12.head(5))
aList = list(cluster12.head(5).index.tolist())

pList = []

for image_id in aList:
    regions = vg.get_region_descriptions_of_image(id=image_id+1)
    for j in range(len(regions)):
        pList.append(regions[j].phrase.split())

    # print("The %s region descriptions is: %s" %(image_id,regions[0].phrase))
    # print("The %s region descriptions is: %s" %(image_id,regions[1].phrase))
    # print("The %s region descriptions is: %s" %(image_id,regions[2].phrase))
    # print("The %s region descriptions is: %s" %(image_id,regions[3].phrase))
    # print("The %s region descriptions is: %s" %(image_id,regions[4].phrase))

pList = sum(pList, [])
pList = Counter(pList)
frqHundred = pList.most_common(100)
print(frqHundred)
