import sys
import numpy as np
import pandas as pd
import torch
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





with open("data/idxIdDict.pickle", "rb") as fr:
    idDict = pickle.load(fr)

#with open("data/query01_0720_2.pickle", "rb") as fr:
with open("data/query01_0720.pickle", "rb") as fr:
    qGraphs = pickle.load(fr)

def graphShowName(nexG):
    #nx.draw(nexG, with_labels=True)
    relabelDict = {}
    for idx in nexG.nodes():
        relabelDict[idx] = nexG.nodes[idx]['name']

    nexG = nx.relabel_nodes(nexG, relabelDict)
    plt.figure(figsize=[5,5])
    nx.draw(nexG,  with_labels=True)
    plt.show()

gList = [
180, 5489, 9127, 4595, 278, 7740, 511, 1993
]
qGList = [0, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ]

[print(idDict[idx]) for idx in gList]
[graphShowName(qGraphs[idx]) for idx in qGList]

sys.exit()



# imgIdList = [1, 399, 593, 594,
#              744, 1505, 631, 1321]
# imgIdList = [2469, 1353, 5301, 1512,
#              2274, 928, 2268, 3336]
# imgIdList = [6579, 1527, 390, 336,
#              1518, 5630, 7483, 1707]
# imgIdList = [3516, 168, 1538, 3461,
#              4584, 5814, 7045, 4309]
imgIdList = [6283, 8315, 6973, 8304,
             6326, 7399, 4131, 5390]

[print(idDict[idx]) for idx in imgIdList]
