import YEmbedding as yed


''' 
   Y data 생성을 위해 image에 대한 text description을 이미지 별로 모음
   jsonpath : './data/region_descriptions.json'
   xlxspath : './data/image_regions.xlsx'
   stopword 삭제하는 코드가 실수로 삭제되었음 < readme 쓰기 위해 일단 생략함 
   실제로 사용한 xlsx는
'''
def jsontoxml(imgCnt, jsonpath, xlsxpath) :
    with open(jsonpath) as file:  # open json file
        data = json.load(file)
        wb = Workbook()  # create xlsx file
        ws = wb.active  # create xlsx sheet
        ws.append(['image_id', 'region_sentences'])
        phrase = []

        q = 0
        for i in data:
            if q == imgCnt:
                break
            regions = i.get('regions')
            imgId = regions[0].get('image_id')
            k = 0
            for j in regions:
                if k == 7:
                    break
                phrase.append(j.get('phrase'))
                k += 1
            sentences = ','.join(phrase)
            ws.append([imgId, sentences])
            phrase = []
            q += 1
        wb.save(xlsxpath)
# # YEmbedding값(labels)을 저장
# ut.jsontoxml(5000, jsonpath, xlxspath1)
# ut.jsontoxml(10000, jsonpath, xlxspath2)



'''label txt 저장'''
# xlxspath = 'data/image_regions.xlsx'
# # Y - image, cluser 몇 번인지~
# embedding_clustering = yed.YEmbedding(xlxspath)
# idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
# label = idCluster['cluster']
# j = label.tolist()
# list_a = list(map(str, j))
#
# '''txt 로 저장'''
# with open('data/cluster.txt', 'w') as file:
#    file.writelines(','.join(list_a))




xlxspath = 'data/image_regions5000.xlsx'
# Y - image, cluser 몇 번인지~
embedding_clustering = yed.YEmbedding(xlxspath)
idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
label = idCluster['cluster']
j = label.tolist()
list_a = list(map(str, j))

'''txt 로 저장'''
with open('data/cluster5000.txt', 'w') as file:
   file.writelines(','.join(list_a))



#
# xlxspath = 'data/image_regions10000.xlsx'
# # Y - image, cluser 몇 번인지~
# embedding_clustering = yed.YEmbedding(xlxspath)
# idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
# label = idCluster['cluster']
# j = label.tolist()
# list_a = list(map(str, j))
#
# '''txt 로 저장'''
# with open('data/cluster10000.txt', 'w') as file:
#    file.writelines(','.join(list_a))