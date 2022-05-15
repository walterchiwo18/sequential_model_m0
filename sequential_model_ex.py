from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(10, input_dim=15))
model.add(Activation('relu'))

model.summary()
