import os
from keras.preprocessing import sequence
from keras.datasets import imdb
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from NLP.PackageGeneratorForLSTM import GeneratorForLSTM, BidirectionalGenerator

data_dir = 'E:/PyCharmProjects/CvD/data/jena_climate'
fname = os.path.join(data_dir,'jena_climate_2009_2016.csv')
old_load = np.load
np.load = lambda *a,**k:old_load(*a,allow_pickle=True,**k)
f = open(fname)
data = f.read()
f.close()



max_features = 10000
maxlen = 500

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)

np.load = old_load
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = sequence.pad_sequences(x_test,maxlen=maxlen)

model = Sequential()
model.add(layers.Embedding(max_features,128))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size=1024,
                    validation_split=0.2)


acc = history.history['acc']

val_acc = history.history['val_acc']

epochs = range(1,len(acc)+1)
plt.figure()

plt.plot(epochs,acc,'bo')
plt.plot(epochs,val_acc,'b')
plt.show()