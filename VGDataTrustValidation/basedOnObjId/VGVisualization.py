from visual_genome import api as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
import json
import sys
import time


try:
    from StringIO import StringIO as ReadBytes
except ImportError:
    print("Using BytesIO, since we're in Python3")
    #from io import StringIO #..
    from io import BytesIO as ReadBytes

image_id = 1
#data load
image = vg.get_image_data(image_id)

with open('../data/scene_graphs.json') as file:  # open json file
    sceneJson = json.load(file)

# 0번 scenegraph의
relationships = sceneJson[0]['relationships']
objects = sceneJson[0]['relationships']

print("sceneJson image_id : ", sceneJson[0]['image_id'])
print('relationship_id : ', relationships['relationship_id'])
print('object_id : ', relationships[0]['object_id'])
print('subject_id: ', relationships[0]['subject_id'])

object_id = relationships[0]['object_id']
subject_id = relationships[0]['subject_id']


with open('../data/objects.json') as file:  # open json file
    objects = json.load(file)


print('objects / object_id / object_name')
print('object_id : ',objects[0]['object_id'])
print('names : ', objects[0]['names'])

print("object_id x:%d, y:%d, width:%d, height:%d"
      % (regions[0].x, regions[0].y, regions[0].width, regions[0].height))

print('object_id : ',objects[1]['object_id'])
print('names : ', objects[1]['names'])






# with open('./data/objects.json') as file:  # open json file
#     objects = json.load(file)
#
# print("sceneJson image_id : ", objects[0]['image_id'])
# print('object_id : ', objects[0]['objects'][0]['object_id'])
# print('object_id : ', objects[0]['objects'][0]['names'])
# print('object_id : ', objects[0]['objects'][1]['object_id'])
# print('object_id : ', objects[0]['objects'][1]['names'])
# print('object_id : ', objects[0]['objects'][2]['object_id'])
# print('object_id : ', objects[0]['objects'][2]['names'])



sys.exit()


#region_description load
regions = vg.get_region_descriptions_of_image(id=image_id)
# object loaction with phrase in first phrase of regions
print("The first region descriptions is: %s" % regions[0].phrase)
print("It is located in a bounding box specified by x:%d, y:%d, width:%d, height:%d"
      % (regions[0].x, regions[0].y, regions[0].width, regions[0].height))

# #region_description print
# for i, region in enumerate(regions):
#     print("regions[%2d].id=%7d .phrase='%s'" % (i, region.id, region.phrase,))


#visualize with bounding box
def visualize_regions(image, regions):
    response = requests.get(image.url)
    img = PIL_Image.open(ReadBytes(response.content))
    plt.imshow(img)
    ax = plt.gca()
    for region in regions:
        ax.add_patch(Rectangle((region.x, region.y),
                               region.width, region.height,
                               fill=False,
                               edgecolor='red', linewidth=3))
        ax.text(region.x, region.y, region.phrase, style='italic',
                bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    #plt.tick_params(labelbottom='off', labelleft='off')
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.show()

#visualize on plt
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

visualize_regions(image, regions[31:38])