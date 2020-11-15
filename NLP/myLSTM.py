from keras.layers import LSTM,Embedding, Dense
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np

max_features = 10000
maxlen = 1000

print('Loading data...')
old_load = np.load
np.load = lambda *a,**k:old_load(*a,allow_pickle=True,**k)
(input_train,y_train),(input_test,y_test) = imdb.load_data()
np.load = old_load

print((input_train))

print('Pad sequence (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen = maxlen)
input_test = sequence.pad_sequences(input_test,maxlen= maxlen)

model = Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(input_train,y_train,
          epochs = 10,
          batch_size = 2048,
          validation_split = 0.2)


acc = history.history['acc']
loss = history.history['loss']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo')
plt.plot(epochs,val_acc,'b')
plt.figure()

plt.plot(epochs,loss,'bo')
plt.plot(epochs,val_loss,'b')

plt.show()