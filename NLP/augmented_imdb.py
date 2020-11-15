import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense
import numpy as np
import codecs
import matplotlib.pyplot as plt

imdb_dir = 'E:/PyCharmProjects/CvD/data/imdb/aclImdb'
glove_dir = 'E:/PyCharmProjects/CvD/data/glove/'
train_dir = os.path.join(imdb_dir,'train')
test_dir = os.path.join(imdb_dir,'test')
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

embedding_index = {}
labels = []
texts = []

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

def pre_trin():
    global labels
    for lable_type in ['neg','pos']:
        dir_name = os.path.join(train_dir,lable_type)
        for fname in os.listdir(dir_name):
            if fname[-4:]=='.txt':
                f = codecs.open(os.path.join(dir_name,fname),'r','utf_8_sig')
                texts.append(f.read())
                f.close()
                if lable_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    print('Labels and text has been prepared. Proceeding to learning...')

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.'%len(word_index))

    data = pad_sequences(sequences,maxlen=maxlen)

    labels = np.asarray(labels)
    print('Shape of data tensor is:',data.shape)
    print('Shape of label tensor is:',labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]

    x_val = data[training_samples: training_samples+validation_samples]
    y_val = labels[training_samples: training_samples+validation_samples]

    f = codecs.open(os.path.join(glove_dir,'glove.6B.100d.txt'),'r','utf_8_sig')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print('Found %s word vectors.' %len(embedding_index))

    embedding_dim = 100

    embedding_matrix = np.zeros((max_words,embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    print('Words were loaded from glove file. Proceeding to building the model...')
    model = Sequential()
    model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train,y_train,
                        epochs=100,
                        batch_size=64,
                        validation_data=(x_val,y_val))

    model.save('pre_trained_glove_model.h5')

    print('Graphs:')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.plot(epochs,acc,'bo',label='Tr acc')
    plt.plot(epochs,val_acc,'b',label='Val acc')
    plt.legend()

    plt.figure()
    plt.plot(epochs,loss,'bo',label='Train loss')
    plt.plot(epochs,val_loss,'b',label='Val loss')
    plt.legend()

    plt.show()

for lable_type in ['neg','pos']:
    dir_name = os.path.join(test_dir,lable_type)
    for fname in os.listdir(dir_name):
        if fname[-4:]=='.txt':
            f = codecs.open(os.path.join(dir_name,fname),'r','utf_8_sig')
            texts.append(f.read())
            f.close()
            if lable_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences,maxlen=maxlen)
y_test = np.asarray(labels)
model = Sequential()
model.add(Embedding(max_words, 100, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])


model.load_weights('pre_trained_glove_model.h5')
score = model.evaluate(x_test,y_test)

print(score)