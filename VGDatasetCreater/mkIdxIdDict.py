


with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)

'''Dictionary, idx(0:100,000) : Image Id'''
data = data[:100000]

idxIdDict = {idx+1: data[idx]["image_id"] for idx, imageInfo in enumerate(data)}  #

with open("data/idxIdDict.pickle", "wb") as fw:  # < node[nId]['attr'] = array(float)
    pickle.dump(idxIdDict, fw)





import sys
import json
import pickle


with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)

with open("data/idxIdDict.pickle", "rb") as fr:
    idxIdDict = pickle.load(fr)

# print(idxIdDict[1000])
print(idxIdDict[99999])
print(idxIdDict[100000])

#json data는 0번부터 시작 -> data[89] : Image Id 90 번에 대한 objects

print(data[99998]['objects'])
print(data[99999]['objects'])


