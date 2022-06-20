import networkx as nx
import pickle
import matplotlib.pyplot as plt



with open("./data/networkx_sifted.pickle", "rb") as fr:
    data = pickle.load(fr)




gId = 6
gI = data[gId]
print(data)
print('data[gId] : ', data[gId])





plt.figure(figsize=[15, 7])
nx.draw(gI, with_labels=True)
plt.show()
