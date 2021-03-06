from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding
import numpy as np

max_features = 10000
maxlen = 20

np_load_old = np.load

np.load = lambda *a,**k:np_load_old(*a,allow_pickle=True,**k)
(x_train,y_train),(x_test,y_test)  = imdb.load_data(num_words=max_features)

np.load = np_load_old

x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)

model=Sequential()
model.add(Embedding(1000,8,input_length=maxlen))

model.add(Flatten())

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.summary()

history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)