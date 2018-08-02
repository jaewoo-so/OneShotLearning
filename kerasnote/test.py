# from keras.applications import mobilenet
from keras import layers , losses , optimizers , models 
from functools import *
from keras import  utils
from keras import datasets


class SimpleModel:
    def __init__(self):
        self.l1 = layers.Conv2D(64 , (3,3) , activation = 'elu' , padding = 'same')
        self.l2 = layers.MaxPooling2D((2,2))
        self.l3 = layers.BatchNormalization()
        self.l4 = layers.Conv2D(32 , (3,3) , activation = 'elu' , padding = 'same')
        self.l5 = layers.MaxPooling2D((2,2))
        self.l6 = layers.BatchNormalization()
        self.l7 = layers.Flatten()
        self.l8 = layers.Dense(32 , activation = 'relu' )
        self.l8 = layers.Dense(10 , activation = 'softmax' )
        self.SetLayerList()
        
    def SetLayerList(self):
        self.layers = [ x[-1] for x in self.__dict__.items()]
        
    def Build(self,input):
        alllist = [input ] + self.layers
        net = reduce(lambda f,s : s(f),alllist)
        return net

temp = SimpleModel()
temp2 = temp.__dict__.items()

#(x,y),(xt,yt) = datasets.mnist.load_data()
#y = utils.to_categorical(y,10)


#input1 = layers.Input(x.shape[1:])
input1 = layers.Input((23,))

output = temp.Build(input1)
model = models.Model(input1 , output)