import myTools

import theano
import numpy
from PIL import Image

from os import listdir
from os.path import isfile, join

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataSet=myTools.loadImages('/home/athanasiostsiaras/Downloads/images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)
	
plt.show(plt.imshow(dataSet[0][0], cmap=cm.binary))

masks=myTools.loadImages('/home/athanasiostsiaras/Downloads/masks', 819, 819, 1)

plt.show(plt.imshow(masks[0][0], cmap=cm.binary))


