import myTools
import myClasses
import theano
import numpy
import pylab
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import sys
import math
import random
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import theano.tensor as T
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
#from nolearn.lasagne import NeuralNet
#from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn
from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, MaxPool2DLayer, InverseLayer

argLR=float(sys.argv[1])

argWD=float(sys.argv[2])

argEpochs=int(sys.argv[3])

print('Learning rate: %f' % (argLR))
print('Weight decay: %f' % (argWD))
print('Number of epochs: %d' % (argEpochs))

random.seed(123)

### DATASET
myTifs=Image.open('../../ISBIdata/train-volume.tif')

dataSet=np.array(np.zeros(shape=(30, 1, 512, 512)))

for i in range(30):
    try:
        myTifs.seek(i)
        dataSet[i][0]=myTifs
    except EOFError:
        # Not enough frames in img
        break

dataSet=dataSet.astype(numpy.uint8)

dataSet=myTools.augmentData(dataSet, numOfTiles=1, overlap=False, imageWidth=512, imageHeight=512)#830 850

dataSet=dataSet.astype(numpy.float32)


imageMeans=numpy.mean(dataSet, axis=(1,2,3), keepdims=True)


dataSet=dataSet-imageMeans


### MASKS
myTifsMasks=Image.open('../../ISBIdata/train-labels.tif')

masks=np.array(np.zeros(shape=(30, 1, 512, 512)))

for i in range(30):
    try:
        myTifsMasks.seek(i)
        masks[i][0]=myTifsMasks
    except EOFError:
        # Not enough frames in img
        break

masks[masks>0]=1

masks=masks.astype(numpy.float32)
#masks=myTools.dt(masks, 20)

masks=myTools.augmentData(masks, numOfTiles=1, overlap=False, imageWidth=512, imageHeight=512)

masks=masks.astype(numpy.float32)


### DATASET SPLIT
splitPoint=int(math.floor(dataSet.shape[0]*0.9))

train=dataSet#dataSet[0:splitPoint, :, :, :]
test=dataSet[splitPoint+1:dataSet.shape[0], :, :, :]

dataSet=None


imgsWidth, imgsHeight =train[0][0].shape

data_size=(None,1,imgsWidth,imgsHeight)

#43 for 172 images. 57 for 229 images - to retain 4 image batch
numOfBatches=60
batchSize=int(math.floor(train.shape[0]/numOfBatches))


myNet=myTools.createNN(data_size, X=train, Y=masks, valX=test, valY=masks[splitPoint+1:masks.shape[0], :, :, :], epochs=argEpochs, n_batches=numOfBatches, batch_size=batchSize, learning_rate=argLR, w_decay=argWD)


#res=myNet(test[0:2, :, :, :])
res1=myNet(np.reshape(test[0,:,:,:], (1,1,test.shape[2],test.shape[3])))
res2=myNet(np.reshape(test[1,:,:,:], (1,1,test.shape[2],test.shape[3])))

#print('Total cost on test set: %f' % (myTools.myTestCrossEntropy(res,masks[splitPoint+1:masks.shape[0], :, :, :])))


numpy.save("outfile1.npy", res1[0][0])
numpy.save("outfile2.npy", res2[0][0])
