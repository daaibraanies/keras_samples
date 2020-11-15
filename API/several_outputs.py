from keras import Input, layers,Sequential,Model

vocab_size = 50000
num_income_group = 10

posts_input = Input(shape=(None,), dtype='int32',name='posts')
embedded_posts = layers.Embedding(256,vocab_size)(posts_input)
x = layers.Conv1D(128,5,activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1,name='age')(x)
income_prediction = layers.Dense(num_income_group,
                                 activation='softmax',
                                 name='income')(x)
gender_prediction = layers.Dense(1,activation='sigmoid',name='gender')(x)

model = Model(posts_input,
              [age_prediction,income_prediction,gender_prediction])

model.compile(optimizer='rmsprop',
              loss=['mse','categorical_crossentropy','binary_crossentropy'],
              loss_weights=[0.25,1.,10.])