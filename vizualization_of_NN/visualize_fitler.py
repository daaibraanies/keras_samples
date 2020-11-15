from keras.applications import VGG16
from keras import backend as bk
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def deprocess_image(x):
    x-=x.mean()
    x /= (x.std()+1e-5)
    x*=0.1
    x+=0.5
    x=np.clip(x,0,1)
    x*=255
    x=np.clip(x,0,255).astype('uint8')
    return x


model = VGG16(weights='imagenet',
              include_top=False)


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = bk.mean(layer_output[:, :, :, filter_index])

    grads = bk.gradients(loss,model.input)[0]
    grads /= bk.sqrt(bk.mean(bk.square(grads)+1e-5))

    iterate = bk.function([model.input],[loss,grads])
    input_img_data = np.random.random((1, size, size, 3))*20+128.

    step = 1.
    for i in range(40):
        loss_value, grad_value = iterate([input_img_data])
        input_img_data += grad_value*step

    img = input_img_data[0]
    return deprocess_image(img)

layer_name = 'block4_conv1'
size = 64
margin = 5


results = np.zeros((8*size+7*margin, 8*size+7*margin, 3))

for i in range(8):
    for j in range(8):
        filter_index = i+(j*8)
        filter_img = generate_pattern(layer_name, filter_index, size=size)

        horizontal_s = i*size + i*margin
        horizontal_e = horizontal_s+size

        vertical_s = j*size + j*margin
        vertical_e = vertical_s + size

        results[horizontal_s: horizontal_e, vertical_s:vertical_e, :] = filter_img


plt.imshow(results)
plt.show()
#fig = plt.figure(figsize=(8,8))
#columns = 8
#rows = 8

#for i in range(1,columns*rows+1):
#    img = generate_pattern(layer_name,i-1,size=size)
#    fig.add_subplot(rows,columns,i)
#    plt.imshow(img)

#plt.show()
