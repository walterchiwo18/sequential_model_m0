from keras.models import Sequential
from keras.layers import Dense, Activation
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot

model = Sequential()

model.add(Dense(10, input_dim=15))
model.add(Activation('relu'))

#model.summary()

SVG(model_to_dot(model).create(prog='dot',format='avg'))
