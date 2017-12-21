import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import numpy
import numpy as np
import sys
import myTools
import scipy
from tifffile import imsave
from skimage.io._plugins import freeimage_plugin as fi
from libtiff import TIFFfile, TIFFimage

#set output filename
outName=sys.argv[1]
#get model filename
modelFilename=sys.argv[2]
#get number of filters
numberOfFilters=sys.argv[3]
#open the images
myTifs=Image.open('../../ISBIdata/test-volume.tif')


data = numpy.random.rand(5, 301, 219)

images=np.array(np.zeros(shape=(30, 1, 512, 512)))

results=np.zeros((30, 512, 512))

for i in range(30):
    try:
        myTifs.seek(i)
        images[i][0]=myTifs
    except EOFError:
        # Not enough frames in img
        break


#plt.show(plt.imshow(images[0][0]))\
for j in range(30):

	image=images[j, :, :, :]
	image=image.reshape(1, 1, images.shape[2], images.shape[3])

	image=image.astype(numpy.uint8)
	image=myTools.augmentData(image, numOfTiles=1, overlap=False, imageWidth=512, imageHeight=512)
	image=image.astype(numpy.float32)
	imageMeans=numpy.mean(image, axis=(1,2,3), keepdims=True)


	image=image-imageMeans
	#setting parameters for the network
	data_size=(None,1,images[0][0].shape[0],images[0][0].shape[1])
	#load the pretrained network
	myNet=myTools.createPretrainedNN2(data_size, modelFile=modelFilename, filters=numberOfFilters)
	#make predictions for the image
	res=myNet(image)
	#results[j, :, :]=res[0][0]
	scipy.misc.imsave('../../ISBIdata/areYouKiddingMe/res'+str(j)+'.png', res[0][0])


#scipy.misc.imsave('../../ISBIdata/res'+str(j)+'.jpg' tempPrediction[0][0])
