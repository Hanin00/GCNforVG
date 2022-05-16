import YEmbedding as yed
#
# '''label txt 저장'''
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




xlxspath = 'data/image_regions10000.xlsx'
# Y - image, cluser 몇 번인지~
embedding_clustering = yed.YEmbedding(xlxspath)
idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
label = idCluster['cluster']
j = label.tolist()
list_a = list(map(str, j))

'''txt 로 저장'''
with open('data/cluster10000.txt', 'w') as file:
   file.writelines(','.join(list_a))