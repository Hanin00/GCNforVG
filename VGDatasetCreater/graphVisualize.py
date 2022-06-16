import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image
import time
import pandas as pd
import pickle
import sys


with open("./data/networkx1000_new.pickle", "rb") as fr:
    data = pickle.load(fr)


gI = data[5]
print(gI)
plt.figure(figsize=[15, 7])
nx.draw(gI, with_labels=True)
plt.show()