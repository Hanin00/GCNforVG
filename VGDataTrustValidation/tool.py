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




