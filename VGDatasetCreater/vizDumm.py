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



# with open("data/v3_x100/v3_x1000.pickle", "rb") as fr:
#     v3X1000 = pickle.load(fr)

# with open("data/query01_0720.pickle", "rb") as fr:
#     qG = pickle.load(fr)

def graphShowName(nexG):
    #nx.draw(nexG, with_labels=True)
    relabelDict = {}
    for idx in nexG.nodes():
        relabelDict[idx] = nexG.nodes[idx]['name']

    nexG = nx.relabel_nodes(nexG, relabelDict)
    plt.figure(figsize=[5, 5])
    nx.draw(nexG,  with_labels=True)
    plt.show()



# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)

with open("data/idxIdDict.pickle", "rb") as fr:
    idDict = pickle.load(fr)


with open("data/query01_0720.pickle", "rb") as fr:
    qQ = pickle.load(fr)


qList = [0, 2, 10, 12, 15 ]
[graphShowName(qQ[idx]) for idx in qList]

sys.exit()



# imgIdList = [1, 399, 593, 594,
#              744, 1505, 631, 1321]
# imgIdList = [2469, 1353, 5301, 1512,
#              2274, 928, 2268, 3336]
imgIdList = [6579, 1527, 390, 336,
             1518, 5630, 7483, 1707]
imgIdList = [3516, 168, 1538, 3461,
             4584, 5814, 7045, 4309]
imgIdList = [6283, 8315, 6973, 8304,
             6326, 7399, 4131, 5390]

[print(idDict[idx]) for idx in imgIdList]
