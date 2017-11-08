from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2

# prebuild model with the pre-trained weights on imagenet
model = VGG16(weights = 'imagenet', include_top = True)
sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy')

# resize the input image into the format which is accepted by the
# VGG16 model
im = cv2.imread('train.jpg')
im = cv2.resize(im, (224, 224))
im = np.expand_dims(im, axis = 0)


# prediction time :D
out = model.predict(im)
plt.plot(out.ravel())
plt.show()
print(np.argmax(out))
