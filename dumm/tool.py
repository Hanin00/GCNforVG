# install NLTK Data
# ref) https://www.nltk.org/install.html
from visual_genome import api as vg
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import json
import nltk
from collections import Counter
from nltk.corpus import conll2000
import sys
import time


def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

conllDict = {word: tag for word, tag in conll2000.tagged_words(tagset='universal')}

print(conllDict['the'])


with open("./data/synsetDict_1000.pickle", "rb") as fr:
    synDict2 = pickle.load(fr)

with open("./data/synsetDictV3.pickle", "rb") as fr:
    synDict3 = pickle.load(fr)

print(get_key(synDict2,'the'))
print(get_key(synDict3,'the'))



sys.exit()


conllDict = {word: tag for word, tag in conll2000.tagged_words(tagset='universal')}


def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"


def vGphShow(nexG):
    #nx.draw(nexG, with_labels=True)

    plt.figure(figsize=[15, 7])
    nx.draw(nexG,  with_labels=True)
    plt.show()


def graphShowId(nexG):
    #nx.draw(nexG, with_labels=True)
    relabelDict = {}
    for idx in nexG.nodes():
        relabelDict[idx] = nexG.nodes[idx]['originId']

    nexG = nx.relabel_nodes(nexG, relabelDict)
    plt.figure(figsize=[15, 7])
    nx.draw(nexG,  with_labels=True)
    plt.show()

def graphShowName(nexG):
    #nx.draw(nexG, with_labels=True)
    relabelDict = {}
    for idx in nexG.nodes():
        relabelDict[idx] = nexG.nodes[idx]['name']

    nexG = nx.relabel_nodes(nexG, relabelDict)
    plt.figure(figsize=[15, 7])
    nx.draw(nexG,  with_labels=True)
    plt.show()




