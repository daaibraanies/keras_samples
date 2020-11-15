import keras
import os
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
import tensorflow as tf
from keras.utils import plot_model

print('test')
#################################
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#config = tf.ConfigProto(device_count = {'CPU':0})
#sess = tf.Session(config)
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#################################

batch_size = 1024

with tf.device('/gpu:0'):
    embedding_metadata = {'embed':'embed'}
    max_features = 2000
    max_len = 500

    old_load = np.load

    np.load = lambda *a,**k:old_load(*a,allow_pickle = True,**k)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    np.load = old_load

    x_train = sequence.pad_sequences(x_train,maxlen=max_len)
    x_test = sequence.pad_sequences(x_test,maxlen=max_len)

    model = keras.models.Sequential()

    model.add(layers.Embedding(max_features,128,input_length=max_len,name='embed'))
    model.add(layers.Conv1D(32,7,activation='relu'))
    model.add(layers.MaxPool1D(5))
    model.add(layers.Conv1D(32,7,activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(1))

    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir='tb_log_dir',
            histogram_freq=1,
            embeddings_freq=1,
        )
    ]

#plot_model(model, to_file='model.png',show_shapes=True)

history = model.fit(x_train,y_train,
                        epochs=10,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=callbacks)
