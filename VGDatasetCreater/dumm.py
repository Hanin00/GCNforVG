import networkx as nx
import pickle
import matplotlib.pyplot as plt
import json
import nltk
from nltk.corpus import conll2000
import sys
import time

from collections import Counter







# '''
#     2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
#         -> noun이 두 개 일 경우 기존 synset List에서 가장 많이 사용되는 단어를 사용하고, 개수가 일치 할 경우 [0]의 단어를 name으로 함
#         -> NLTK로 분할한 noun들이 기존 synset이 아닌 경우 전체를 통째로 synset으로 만들기
# '''
# def extractNoun(noun, synsDict, synNameCnter) :
#     conllDict = {word: tag for word, tag in conll2000.tagged_words(tagset='universal')}
#
#     words = noun.split(' ')
#     # noun 판별
#     nouns = []
#     for i in words :
#         try:
#             if conllDict[i] == 'NOUN':
#                 nouns.append(i)
#         except :
#             continue
#     # synset에 해당되는 noun이 있는지 판별
#     nInSynsDictList = []
#     if len(nouns) != 0 :
#         for i in nouns :
#             try :
#                 nInSynsDictList.append(synsDict[i])
#             except :
#                 continue
#
#     # synset에 해당되는 noun 중 언급이 많은 단어 선택
#     name = ''
#     cnt = 0
#     if len(nInSynsDictList) != 0 :
#         for i in nInSynsDictList:
#             if cnt < synNameCnter[i] :
#                 name = i
#                 cnt = synNameCnter[i]
#     else :
#         name = "_".join(sorted(nouns))
#     return name
#
#
#
# def get_key(dict, val):
#     for key, value in dict.items():
#         if val == value:
#             return key
#     return "key doesn't exist"
#
#
# '''
#     1. synset 유무 확인 -> 해당 name을 가진 다른 ObjId가 synset을 가지면 해당 ObjId의 synset 사용 / 없으면 name 확인 후 2. 로
#     2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
#         -> noun이 두 개 일 경우 기존 synset List에서 가장 많이 사용되는 단어를 사용하고, 개수가 일치 할 경우 [0]의 단어를 name으로 함
#         -> NLTK로 분할한 noun들이 기존 synset이 아닌 경우 전체를 통째로 sysset으로 만들기
#             1. nltk로 분할 하기 전 name
#             2. nltk로 추출한 noun을 이어붙여 하나의 단어로
#                 a. people t-shirt
#                 b. people_tshirt
#             3. noun 값 중 하나
#                 a. sorted 한 nounlist[0]
#                 b. nounlist[0]
# '''
#
# gList = []
# imgCnt = 10
# start = time.time()
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
#
# '''synset Name Dict'''
# '''
#     1. synset 유무 확인 -> 해당 name을 가진 다른 ObjId가 synset을 가지면 해당 ObjId의 synset 사용 / 없으면 name 확인 후 2. 로
# '''
# synsetList = []
# originIdList = []
# nonSysnNameList = []
# nonSysnIdList = []
#
# originDict = {}
#
# for ImgId in range(imgCnt) :
#     imageDescriptions = data[ImgId]["objects"]
#     for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
#         oId = imageDescriptions[j]['object_id']
#         try :
#             synsetName = imageDescriptions[j]['synsets'][0].split(".")
#             synsetList.append(synsetName[0])
#             originIdList.append(str(oId))
#             originDict[str(oId)] = imageDescriptions[j]['names'][0]
#
#         except Exception  :
#             nonSysnNameList.append(imageDescriptions[j]['names'][0])
#             nonSysnIdList.append(str(oId))
#
#
# synNameCnter = Counter(synsetList)
#
#
# '''
#     1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우, objId로 synsDict에서 synsetName을 찾음
#     2. nonSynsetName/Id에서 해당 원소를 제외하고, synsDict에 추가함(objId : objName(synset))
#
#     없는 경우, 2로 넘어감
# '''
#
# synsDict = {idx: name for idx, name in zip(originIdList, synsetList)}
# nonSynsDict = {name: value for name, value in zip(nonSysnIdList, nonSysnNameList)}
#
# for i in range(len(nonSysnNameList)) :
#     try :
#         sameNameId = get_key(originDict, nonSysnNameList[i]) # 1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우,
#         synsDict[nonSysnIdList[i]] = synsDict[sameNameId]
#
#     except :
#             #todo 2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
#         name = extractNoun(nonSysnNameList[i], synsDict, synNameCnter)
#         synsDict[nonSysnIdList[i]] = name
