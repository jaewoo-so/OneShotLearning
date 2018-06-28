from keras import layers 
from keras import models , optimizers , losses
from keras import initializers
import numpy as np
from keras import backend as K

x = [[1,2,3],[4,5,6]]
xx =[[10,20,30],[40,50,60]]
x = np.asarray(x)
xx = np.asarray(xx)
y = [0,1]

input1 = layers.Input(shape = (3,) )
input2 = layers.Input(shape = (3,) )

l = layers.Dense( 1 , kernel_initializer = initializers.RandomUniform(0,1) )
rel =layers.Lambda(lambda x: K.reverse(x,axes=0))
rel2 = layers.Lambda(lambda x : K.map_fn()) 
x1 = l(input1)
x2 = l(input2)
x3 = rel(x2)


#model0 = models.Model(inputs = input1, outputs = x1)
#model0.compile(loss = losses.binary_crossentropy , optimizer = optimizers.Adam())
#res = model0.predict(x)

merged = layers.merge.Subtract()([x2 , x1])

model1 = models.Model(inputs = [input1 , input2] , outputs = [x1,x2])
model1.compile(loss = losses.binary_crossentropy , optimizer = optimizers.Adam())
res1 = model1.predict([x , xx])

model2 = models.Model(inputs = [input1 , input2] , outputs = merged)
model2.compile(loss = losses.binary_crossentropy , optimizer = optimizers.Adam())
res2 = model2.predict([x , xx])

model3 = models.Model(inputs = [input1 , input2] , outputs = [merged , x3])
model3.compile(loss = losses.binary_crossentropy , optimizer = optimizers.Adam())
res3 = model3.predict([x , xx])
print(res3[0])
print()
print(res3[1])
print()
print("-"*30)

print(res1)
print()
print(res2)
