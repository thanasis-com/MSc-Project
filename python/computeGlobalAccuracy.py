import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import numpy
import numpy as np
import sys
import myTools
import scipy
from os import listdir
from os.path import isfile, join
from sklearn.metrics import mean_squared_error, accuracy_score, jaccard_similarity_score, roc_auc_score
from math import sqrt

#get the produced masks' path
imagePath=sys.argv[1]

#get all images' names
filenames = [f for f in listdir(imagePath) if isfile(join(imagePath, f))]

jaccardList=list()
aucList=list()

for f in filenames:
	#open the image
	myMask=Image.open(imagePath + '/' + f, 'r')
	#image as numpy array
	myMask=numpy.asarray(myMask).copy().flatten()
	#myMask=myTools.cropCenter1(myMask, 100)
	#print(myMask)
	#sys.exit()
	myMask[myMask<128]=0
	myMask[myMask>=128]=1
	#plt.show(plt.imshow(myMask))
	#print(myMask)

	#open the expert mask
	theMask=Image.open('../../masksExpertBigTest/' + '/' + f, 'r')
	#image as numpy array
	theMask=numpy.asarray(theMask).copy().flatten()

	theMask[theMask<=50]=0
	theMask[theMask>50]=1
	#plt.show(plt.imshow(theMask))
	#print(theMask)
	#sys.exit()
	thisJaccard = jaccard_similarity_score(theMask, myMask)
	thisAUC = roc_auc_score(theMask, myMask)

	jaccardList.append(thisJaccard)
	aucList.append(thisAUC)


print('Average Jaccard Index: ' + str(numpy.mean(jaccardList)) + '\n')
print('Average AUC: ' + str(numpy.mean(aucList)) + '\n')
