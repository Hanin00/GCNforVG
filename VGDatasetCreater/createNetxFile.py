import sys
import numpy as np
import pandas as pd
import torch
import csv

import torch_geometric.utils

import util as ut
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import pickle

'''
    Ex edge per 1Img
'''
# objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(1)
# df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, "edgeId": edgeId,
#        "weight": weight, "relatiohship": relatiohship, })
# print(df_edge.head(5))

''' 
    Ex node per 1Img
'''
# objectIds, objectNames = ut.AllNodes(1)
# df_node = pd.DataFrame({"objectIds": objectIds, "objectNames": objectNames})
# print(df_node.head(5))


'''
    feature Matrix list(1000,len(objectName),10) 형태로 저장하기
'''
# imgCnt = 1200
# features = []
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
#
# # # fasttext 240마다 멈춰서
# igList = [50, 60, 156, 241, 284, 299, 317, 371, 403, 432, 512, 520, 647, 677, 745, 867, 930, 931, 1102, 1116, 1136,
#           1174, 1196]
# for j in tqdm(range(1, imgCnt)):
#     if j in igList:
#         continue
#     else:
#         objectIds, objectNames = ut.AllNodes(data, j)
#         objDict = {name: value for name, value in zip(objectIds, objectNames)}
#         embDict = ut.FeatEmbeddPerImg(objectNames)
#         features.append(torch.tensor(ut.FeatureMatrix(objectIds, objDict, embDict)))
#
# featM = np.array(features)
# with open("./data/featMatrix1000_1.pickle", "wb") as fw:
#     pickle.dump(featM[:1000], fw)
#
# with open("./data/featMatrix1000_1.pickle", "rb") as fr:
#     data = pickle.load(fr)
#
# data = torch.LongTensor(data)
# print(type(data))
# print(data[0])

# print(data.size)
# print(data[0].shape)
# print(type(data[0]))

'''
    NetworkX 객체 리스트 생성 - 1000개의 이미지에 대한 ObjId, SubjId의 realationship에 대한 Graph 객체 생성
    -> 총 1000개의 undirected Graph
    size (1000, [ObjId 개수])
    
    >> AddEdges 사용
    AllNodes를 통해 중복이 제거된 ObjId와 ObjNameList를 이용해 FeatEmbeddingPerImg에서 Dict를 얻어
    각 노드에 feature(fastText Embedding 값)를 node의 ['attr']로 부여함
    이때 각 fastText Embddding 값의 type은 torch.float32형
    
      networkX 생성 -> featureMatrix를 fasttext로 만드는데 objname이 너무 적어서인지 오류나던 이미지 제외하고 생성
    weight 값 default=1
    node id 그대로 <- feature 반영 할 때 필요할까봐?
    
    networkX 생성 
    -> graph 마다 node id 0으로 시작하도록 변경     
'''
gList = []
imgCnt = 1200
start = time.time()
with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
end = time.time()
print(f"파일 읽는데 걸리는 시간 : {end - start:.5f} sec") # 파일 읽는데 걸리는 시간 : 24.51298 sec
# 이거 아님 igList = [49,59,155,240,283,298,316,370,402,431,511,519,646,676,744,866,929,930,1101,1115,1135,1173,1195]
igList = [50,60,156,241,284,299,317,371,403,432,512,520,647,677,745,867,930,931,1102,1116,1136,1174,1196]

# a = ut.AllEdges(data,393)
# print(a)
# b = ut.AllEdges(data,394)
# print(b)
# c = ut.AllEdges(data,395)
# print(c)

for i in tqdm(range(imgCnt)):
    if i in igList :
        continue
    else :
        objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data,i)
        # networkX graph 객체 생성 ---
        objIdSet, objNameList = ut.AllNodes(data, i)
        df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })
        gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')

        # node attribute 부여 ---
        embDict = ut.FeatEmbeddPerImg(objIdSet,objNameList)
        nx.set_node_attributes(gI, embDict, "attr") # node attribute 부여

        # graph에서 노드 id 0부터 시작하도록 ---
        listA = list(set(objId + subjId))
        listIdx = range(len(listA))
        dictIdx = {name: value for name, value in zip(listA, listIdx)}
        gI = nx.relabel_nodes(gI, dictIdx)
        nx.set_node_attributes(gI, 1, "weight")
        gList.append(gI)

#with open("./data/networkx1000.pickle", "wb") as fw: < node[nId]['attr'] = torch.FloatTensor32
with open("./data/networkx1000_noTensor.pickle", "wb") as fw: #< node[nId]['attr'] = array(float)
    pickle.dump(gList, fw)
    #pickle.dump(gList[:1000], fw)

#with open("./data/networkx1000.pickle", "rb") as fr:
with open("./data/networkx1000_noTensor.pickle", "rb") as fr:
    data = pickle.load(fr)
nx1000 = data[:1000]
print(len(nx1000))
print(nx1000[0].nodes(data=True))
print(type(nx1000[0]))
print(nx1000[0].nodes.data)



# '''dgl.from_networkx test'''
# import dgl
#
# with open("./data/networkx1000.pickle", "rb") as fr:
#     data = pickle.load(fr)
# nx1000 = data[:1000]
#
# G = nx1000[0]
# print(nx1000[0].nodes.data)
# a = dgl.from_networkx(G, node_attrs=['attr', 'weight'])
# print(a)






''' NetworkX 객체에서 Adj 추출하기 - nodeList option으로 줘서 순서에 맞게!!'''
# 굳이 할 필요 없이

# with open("./data/neworkx1000.pickle", "rb") as fr:
#     data2 = pickle.load(fr)

# G = data2[0]
# nx.draw(G, with_labels=True, font_size=5)
# pos = nx.spring_layout(G)
# plt.show()


''' Feature Matrix local 저장 및 Adj local 저장'''
# with open('./data/scene_graphs.json') as file:  # open json file
#     data1 = json.load(file)
#     # objId, objName 불러옴
# errorId = [] # todo relationship에서 Id를 얻어오지 못하는 것을 확인함 -> 해당 Id 기록해서 체크해보기
#
# featMList = []
# adjMList = []
# imgCnt = 1300
# for imgId in tqdm(range(50, imgCnt)):
#     objectIds, objectNames = ut.AllNodes(data1, imgId)
#     # objectId는 중복 X, ordered는 아님
#
#     objId, subjId, relatiohship, edgeId, weight = ut.AllEdges(data1, imgId)
#     objIdList = []
#     objIdList += objId
#     objIdList += subjId
#     objIdList = list(set(objIdList))
#
#     objDict = {name: value for name, value in zip(objectIds, objectNames)}
#     embDict = ut.FeatEmbeddPerImg(objectNames)
#     featM = ut.FeatureMatrix(objIdList, objDict, embDict)
#     featMList.append((featM))
#
#     df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })
#     if(len(df_edge) == 0) :
#         errorId.append(imgId)
#         continue
#     else :
#         G = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')
#         A = nx.adjacency_matrix(G, objIdList)  #
#         # A1 = A.todense()
#         adjMList.append(A)

# print(errorId)

# with open("./data/featMatrix1000.pickle", "wb") as fw:
#     pickle.dump(featMList, fw)
# with open("./data/adjMatrix1000.pickle", "wb") as fw:
#     pickle.dump(adjMList, fw)



'''
    edgeList local 저장 및 확인
'''
# with open("./data/_networkx1000.pickle", "rb") as fr:
#     netx = pickle.load(fr)
#
# import util as ut
# n1, n2 = ut.mkEdgelistPerRange(netx, 1000)
# edgeList = [n1, n2]
#
# with open('./data/edgeList1000.pickle', 'wb') as f:
#     pickle.dump(edgeList[:1001], f)
#
# with open("./data/edgeList1000.pickle", "rb") as fr:
#     edgeList = pickle.load(fr)
#
# print(len(edgeList))
# print(edgeList[0])
# print(len(edgeList[0][0]))
# print(len(edgeList[0][1]))



'''
    edgeList를 tensor로 만들고, (1000, 2, edge 개수)   
'''
#
# def mkEdgeTensor(edgeList) :
#     edgeIdxAll = []
#     for j in range(len(edgeList[0])) :
#         a = edgeList[0]
#         edgeIdxAll.append([torch.tensor(a[j]), torch.tensor(a[j])])
#         #edgeIdxAll.append((torch.tensor(a[0]), torch.tensor(a[1])))
#
#     with open("./data/edgeListTensor.pickle", "wb") as fw:
#         pickle.dump(edgeIdxAll, fw)
#
#     with open("./data/edgeListTensor.pickle", "rb") as fr:
#         data = pickle.load(fr)

# 실행
# with open("./data/edgeList1000.pickle", "rb") as fr:
#     data = pickle.load(fr)
# mkEdgeTensor(data)
# with open("./data/edgeListTensor.pickle", "rb") as fr:
#     data1 = pickle.load(fr)

# print(len(data1)) #1000
# print(len(data1[0])) #2
# print(len(data1[1])) #2
# print(len(data1[0][0])) #31
# print(len(data1[0][1])) #31


'''
    batch 만들어야함 ->
    batch : 어떤 NODE가 어떤 GRAPH에 속하는지 알 수 있도록
    그래프 하나당 노드 개수만큼 그래프 int(Id)를 갖는 리스트    -> 여기서 정확히 ID인지는 잘.. cluster 가 아닌 게 맞나?
    ex. [1,1,1,1,1] : 5 nodes in graph 1 , [2,2,2] : 3 nodes in graph 2
    
    networkx 를 가져와서 node 수 가져옴 -> 빈 리스트에 노드 수만큼 그래프 id를 채움
    
'''
# import networkx as nx
# def mkBatch(netxList) :
#     batchList = []
#     for G in range(len(netxList)) :
#         num = len(nx.nodes(netxList[G]))
#         batch = [G for i in range(num)]  #G로 그래프의 노드 개수인 num만큼 리스트 batch를 초기화
#         batchList.append(torch.tensor(batch))
#     with open("./data/batch.pickle", "wb") as fw:
#         pickle.dump(batchList, fw)
#
#     with open("./data/batch.pickle", "rb") as fr:
#         data = pickle.load(fr)
#
#     return data

# with open("./data/_networkx1000.pickle", "rb") as fr:
#     data = pickle.load(fr)
# data1 = mkBatch(data)
# with open("./data/batch.pickle", "rb") as fr:
#     data1 = pickle.load(fr)

# test
# print(len(data1))
# print(data1[0]) #[0, 0, 0, ... 0, 0, 0, 0]
# print(data1[1]) #[1, 1, 1, ... , 1, 1]
# print(len(nx.nodes(data[1]))) #20
# print(len(data1[1])) #20

'''

    preprocessing_csv file for dgl Graph Classifiaction with dgl  

    graph_edges.csv: containing three columns:
        graph_id: the ID of the graph.
        src: the source node of an edge of the given graph.
        dst: the destination node of an edge of the given graph.
    -> edgeList Tensor 이용해서 만들면 될 듯

    -> feature 값이 안들어가는데..?
    graph_properties.csv: containing three columns:
        graph_id: the ID of the graph.
        label: the label of the graph.
        num_nodes: the number of nodes in the graph.
        ->num_nodes는 G.number_of_nodes()
'''


# #graph_edges.csv ---------------------행열 전환 전 <- zip 안 쓴 이유 : src가 list라 오류나서

# with open("./data/edgeList1000.pickle", "rb") as fr:
#     edgeList = pickle.load(fr)
#
# graphId = range(1000)
# srcList = edgeList[0]
# dstList = edgeList[1]
#
# with open('./data/graph_edges.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['graph_id', 'src','dst'])
#     for graph_id in graphId:
#         for j in range(len(srcList[graph_id])):
#             src = srcList[graph_id][j]
#             dst = dstList[graph_id][j]
#             writer.writerow([graph_id, src, dst])
#             writer.writerow([graph_id, dst, src])


#graph_properties.csv -------------------------------
# with open("./data/clusterSifted1000.pickle", "rb") as fr:
#     labels = pickle.load(fr)
# with open("./data/_networkx1000.pickle", "rb") as fr:
#    networkx = pickle.load(fr)
#
#
# graphId = range(1000)
#
# num_nodes = []
# for G in networkx :
#     num_nodes.append(G.number_of_nodes())
#
# with open('./data/graph_properties.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(graphId)
#     writer.writerow(labels)
#     writer.writerow(num_nodes)
#
# # -------------------------------행열 전환 후 저장
# import pandas as pd
# import numpy as np
# df=pd.read_csv("./data/graph_properties.csv", encoding='cp949')
# df=df.transpose()
# df.to_csv("./data/graph_properties.csv")
# print(df.head())

# -----------------------------------------------------


