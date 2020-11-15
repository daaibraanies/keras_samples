from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt



max_features = 10000
max_len = 500

old_load = np.load
np.load = lambda *a,**k:old_load(*a,allow_pickle=True,**k)

print('Loading data...')

(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words=max_features)
np.load = old_load

print(len(x_train),' train sequence')
print(len(x_test),' test sequence')

print('Pad sequence (samples x time)')
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)

print('x_train shape is ', x_train.shape)
print('x_test shape is ', x_test.shape)


model = Sequential()
model.add(layers.Embedding(max_features,128,input_length=max_len))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train,y_train,
                    epochs = 15,
                    batch_size=1024,
                    validation_split=0.2)

model.save('feedback_class_model_acc84.h5')

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