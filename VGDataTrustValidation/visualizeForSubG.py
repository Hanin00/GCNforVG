# install NLTK Data
# ref) https://www.nltk.org/install.html
from visual_genome import api as vg
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import json
import nltk
from nltk.corpus import conll2000
import sys
import time

def vGphShow(nexG):
    #nx.draw(nexG, with_labels=True)

    plt.figure(figsize=[15, 7])
    nx.draw(nexG,  with_labels=True)
    plt.show()


def graphShow(nexG):
    #nx.draw(nexG, with_labels=True)
    relabelDict = {}
    for idx in nexG.nodes():
        relabelDict[idx] = nexG.nodes[idx]['originId']

    nexG = nx.relabel_nodes(nexG, relabelDict)
    plt.figure(figsize=[15, 7])
    nx.draw(nexG,  with_labels=True)
    plt.show()


with open("./data/networkx_sifted.pickle", "rb") as fr:
    graphs = pickle.load(fr)


gId = 0
G = graphs[gId]
print(G.nodes(data=True))


vGphShow(G)
graphShow(G)



