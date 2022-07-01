import sys
import numpy as np
import util as ut
import json
import pickle
import networkx as nx
from collections import Counter
from nltk.corpus import conll2000
from tqdm import tqdm
import pandas as pd
from visual_genome import api as vg
import matplotlib.pyplot as plt
import time


'''
    이미지 10개 기준으로 object의 synset이 없는 경우나 전처리 되어 사라지는 경우가 있는지 확인   
    기존에는 total embedding시 기존 scene_graph.json에서 object name을 기준으로 출력했는데, 
    ver 3.0.2부터는 만들어진 synset dict의 value 를 기준으로 fasttext embedding 해야함
'''


def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"


# with open("data/test10_syns.pickle", "rb") as fr:
#     syns = pickle.load(fr)
#
# with open("data/test10_total.pickle", "rb") as fr:
#     total = pickle.load(fr)

with open("data/test10.pickle", "rb") as fr:
    data = pickle.load(fr)

for i in data :
    print(data.nodes(data='orig'))
    print(data.nodes(data='name'))




print(data[0])
