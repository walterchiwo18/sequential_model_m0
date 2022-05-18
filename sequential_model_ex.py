from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
#from IPython.display import SVG
#from tensorflow.keras.utils import model_to_dot

model = Sequential()

model.add(Dense(10, input_dim=100,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="rmsprop",loss= "binary_crossentropy", metrics=["accuracy"]) 

#example data
data = np.random.random((1000,100))
labels = np.random.randint(2,size=(1000,1))

model.fit(
    data, 
    labels, 
    batch_size=10, 
    epochs=10, verbose=2, 
    callbacks=None, 
    validation_split=0.2, 
    validation_data=None, 
    shuffle=True, 
    class_weight=None, 
    sample_weight=None, 
    initial_epoch=0)



#model.summary()



#SVG(model_to_dot(model).create(prog='dot',format='avg'))

