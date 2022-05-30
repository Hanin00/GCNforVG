import sys
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
# import pandas as pd
import torch.optim as optim
import pickle
import torch.nn.functional as F
import numpy as np
from gensim.models import FastText
import torch.utils.data as utils
from torch.autograd import Variable

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')


with open("./data/edgeListTensor.pickle", "rb") as fr:
    data = pickle.load(fr)


edges = data
properties = pd.read_csv('./graph_properties.csv')