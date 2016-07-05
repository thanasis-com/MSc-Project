import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from skimage import data, img_as_float, img_as_ubyte
from skimage import exposure
from skimage.transform import rotate

import numpy
import theano
import theano.tensor as T
import scipy
import myTools

import sys
from PIL import Image

masks=myTools.loadImages('../../masks', 819, 819, 1)

for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>0:
             x[...]=0
     else:
	     x[...]=1


#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))
temp=scipy.ndimage.morphology.distance_transform_edt(masks)

for x in numpy.nditer(temp, op_flags=['readwrite']):
     if x>16:
             x[...]=16

Image.fromarray(temp[0][0]).convert('RGB').save('output.png')
#im.save("output.png")
