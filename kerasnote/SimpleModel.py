from keras import layers , losses , optimizers , models 
import numpy as np
from ModelClass import SimpleModel
from functools import *
import keras.backend as K
import cv2
import json

def custom_loss(y_true,y_pred):
    return K.mean((y_true - y_pred)**2)


#region LoadData
samepath = "same.json"
diffpath = "diff.json"

samedict = None 
diffdata = None

with open(samepath , 'r') as f:
    samedict = json.load(f)
with open(diffpath , 'r') as f:
    diffdata = json.load(f)

samedata = samedict['0'] + samedict['1'] + samedict['2'] 
samelabel = np.zeros(len(samedata)).tolist()
difflabel = np.ones(len(diffdata)).tolist()
datas = samedata + diffdata
labels = samelabel + difflabel 
print('datas len = {} , labels len = {}'.format(len(datas) , len(labels)))
#endregion

# Create ImageData
xs = np.asarray(map(lambda x : cv2.imread(x[0]) , datas))
print()