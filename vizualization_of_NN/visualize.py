from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models


test_dir = 'E:/PyCharmProjects/CvD/data/test'
test_cats_dir = os.path.join(test_dir,'cats')
img_path = 'E:/PyCharmProjects/CvD/data/test/cats/cat.1954.jpg'
model = load_model('cvd_small.h5')

img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
img_tensor /= 255.
print(img_tensor.shape)


layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input,outputs= layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]

# names of the layers, so you can have them as a part of your plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# displays the feature maps
for layer_name, layer_activation in zip(layer_names, activations):

    # number of features in the feature map
    n_features = layer_activation.shape[-1]

    # the feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # tiles the activtion channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # tiles each filter into a big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[
                            0, :, :, col * images_per_row + row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(
                channel_image, 0, 255).astype('uint8')

            # displays the grid
            display_grid[
            col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()