import sys
sys.path.append(r'./api/visual_genome_python_driver')
from visual_genome import api as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests

try:
    from StringIO import StringIO as ReadBytes
except ImportError:
    print("Using BytesIO, since we're in Python3")
    #from io import StringIO #..
    from io import BytesIO as ReadBytes


def ImgShow(image) :
    response = requests.get(image.url)
    img = PIL_Image.open(ReadBytes(response.content))
    plt.imshow(img)
    plt.tick_params(labelbottom="Idx : {image}", labelleft=False)
    plt.show()


image_id5 = 6
image1 = vg.get_image_data(id=image_id5)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
ImgShow(image1)   # description을 idx 8번까지만 표시함

sys.exit()

image_id1 = 27
image_id2 = 43
image_id3 = 75
image_id4 = 119
image_id5 = 146
image1 = vg.get_image_data(id=image_id1)
image2 = vg.get_image_data(id=image_id2)
image3 = vg.get_image_data(id=image_id3)
image4 = vg.get_image_data(id=image_id4)
image5 = vg.get_image_data(id=image_id5)

#print("The url of the image is: %s" % image.url)

#egions = vg.get_region_descriptions_of_image(id=image_id)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

def ImgShow(image) :
    response = requests.get(image.url)
    img = PIL_Image.open(ReadBytes(response.content))
    plt.imshow(img)
    plt.tick_params(labelbottom="Idx : {image}", labelleft=False)
    plt.show()

#visualize_regions(image, regions[:])   # description을 idx 8번까지만 표시함
ImgShow(image1)   # description을 idx 8번까지만 표시함
ImgShow(image2)   # description을 idx 8번까지만 표시함
ImgShow(image3)   # description을 idx 8번까지만 표시함
ImgShow(image4)   # description을 idx 8번까지만 표시함
ImgShow(image5)   # description을 idx 8번까지만 표시함

