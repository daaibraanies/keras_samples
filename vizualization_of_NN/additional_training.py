from keras.applications import VGG16
from keras import layers,optimizers,models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt
import os

def smooth_curve(points,factor=0.8):
    smoothed_points=[]

    for point in points:
        if smoothed_points:
            previos = smoothed_points[-1]       ##check what will return
            smoothed_points.append(previos*factor+point*(1-factor))
        else:
            smoothed_points.append(point)

    return smoothed_points

train_dir = 'E:/PyCharmProjects/CvD/data/train'
val_dir = 'E:/PyCharmProjects/CvD/data/validation'
train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')
test_dir = 'E:/PyCharmProjects/CvD/data/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3)
                  )

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

conv_base.trainable = False
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc']
              )

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=25,
    verbose=1
)

print('Showing the right graphs')

plt.plot(50,smooth_curve(history.history['acc']), 'bo')
plt.plot(50,smooth_curve(history.history['val_acc']), 'b')

plt.plot(50,smooth_curve(history.history['loss']), 'bo')
plt.plot(50,smooth_curve(history.history['val_loss']),'b')

plt.show()

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)


test_loss,test_acc = model.evaluate_generator(test_generator, steps=50)
print('test accuracy: ', test_acc)