import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from NLP.PackageGeneratorForLSTM import GeneratorForLSTM, BidirectionalGenerator

data_dir = 'E:/PyCharmProjects/CvD/data/jena_climate'
fname = os.path.join(data_dir,'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines),len(header) - 1))
for i,line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

lookback = 1440
step = 6
delay = 144
batch_size = 4096
print('!!!!!',float_data.shape[-1])
train_gen = GeneratorForLSTM(float_data,
                             lookback,
                             delay,
                             0,
                             200000,
                             shuffle=True,
                             step=step,
                             batch_size=batch_size)


val_gen = GeneratorForLSTM(float_data,
                             lookback,
                             delay,
                             200001,
                             300000,
                             shuffle=True,
                             step=step,
                             batch_size=batch_size)

test_gen = GeneratorForLSTM(float_data,
                             lookback,
                             delay,
                             3000001,
                             None,
                             shuffle=True,
                             step=step,
                             batch_size=batch_size)




val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size


model = Sequential()
model.add(layers.Conv1D(32,3,input_shape=(None,float_data.shape[-1]),activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32,3,activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(),loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=40,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)
plt.figure()

plt.plot(epochs,loss,'bo')
plt.plot(epochs,val_loss,'b')
plt.show()