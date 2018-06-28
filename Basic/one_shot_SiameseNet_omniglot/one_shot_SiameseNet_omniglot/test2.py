from keras import layers
from keras import models , optimizers , losses
from keras import initializers
import numpy as np
from keras import backend as K

def tempfun(x):
    return x * -1.

x = [[1,2,3],[4,5,6]]
xx =[[10,20,30],[40,50,60]]

x = np.asarray(x)
xx = np.asarray(xx)
y = [0,1]

input1 = layers.Input(shape = (3,) )
l = layers.Dense( 2 , kernel_initializer = initializers.RandomUniform(0,1) )(input1)

rel2 = layers.Lambda(lambda x : K.map_fn(tempfun ,x ))(l) 
rel3 = layers.Lambda(lambda x : layers.Dense( 1 , kernel_initializer = initializers.Constant(10) )(x))(l) 


model1 = models.Model(inputs = input1 , outputs = l)
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#res = model1.fit(x,y,epochs = 2 , )

res1 = model1.predict(xx)

print("result")
print(res1)
###
model2 = models.Model(inputs = input1 , outputs = [rel2,rel3])
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#res = model1.fit(x,y,epochs = 2 , )

res2 = model2.predict(xx)

print("result")
print(res2[0])
print()
print(res2[1])

