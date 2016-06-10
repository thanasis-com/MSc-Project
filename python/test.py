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

temp=myTools.loadImages('/home/athanasiostsiaras/Downloads/images', 1024, 1024, 4)

temp1=myTools.dumpAA(temp)
	
plt.show(plt.imshow(temp1[0][0], cmap=cm.binary))

