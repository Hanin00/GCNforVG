import networkx as nx
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

import sys


# with open("./data/featMatrix1000.pickle", "wb") as fw:
#     pickle.dump(featM, fw)
with open("./data/adjMatrix1000.pickle", "rb") as fr:
    data1 = pickle.load(fr)

with open("./data/featMatrix1000.pickle", "rb") as fr:
    data2 = pickle.load(fr)

adjMs = data1[:1000]
featMs = data2[:1000]


print(len(adjMs))
print(len(featMs))

sys.exit()


# gpu 사용
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

