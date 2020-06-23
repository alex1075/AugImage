
import imgaug as ia
ia.seed(1)
# imgaug uses matplotlib backend for displaying images
%matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil

# load images as NumPy arrays and append them to images list
images = []
for index, file in enumerate(glob.glob('images/*.jpg')):
    images.append(imageio.imread(file))
    
# how many images we have
print('We have {} images'.format(len(images)))

# what are the sizes of the images
for index, file in enumerate(glob.glob('images/*.jpg')):
    print('Image {} have size of {}'.format(file[7:], images[index].shape))

# Have a loook at 2 smaller images
ia.imshow(images[2])
ia.imshow(images[7])

# XML file names correspond to the image file names
for index, file in enumerate(glob.glob('images/*.xml')):
    print(file[7:])