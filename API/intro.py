from keras import Input, layers,Sequential,Model

seq_model = Sequential()

seq_model.add(layers.Dense(32,input_shape=(64,),activation='relu'))
seq_model.add(layers.Dense(32,activation='relu'))
seq_model.add(layers.Dense(10,activation='softmax'))

input_tensor = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tensor)
x = layers.Dense(32,activation='relu')(x)
output_tensor = layers.Dense(10,activation='softmax')(x)

model = Model(input_tensor,output_tensor)
model.summary()

