from keras.models import Sequential
from keras.layers import Dense, Activation
#from IPython.display import SVG
#from tensorflow.keras.utils import model_to_dot

model = Sequential()

model.add(Dense(10, input_dim=15,activation='relu'))
model.add(Dense(20, activation='sigmoid'))

model.compile(optimizer="rmsprop",loss= "binary_crossentropy", metrics=["accuracy"]) 
#model.summary()

#SVG(model_to_dot(model).create(prog='dot',format='avg'))

