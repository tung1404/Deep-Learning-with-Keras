from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np


if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    K.set_image_dim_ordering("th")

    # Test pretrained model
    # model = VGG_16('/Users/gulli/Keras/codeBook/code/data/vgg16_weights.h5')
    # model = VGG16(weights='imagenet', include_top=True)
    model = VGG16()
    optimizer = SGD()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    out = model.predict(im)
    print(np.argmax(out))
    
    from keras.applications.vgg16 import decode_predictions
    # convert the probabilities to class labels
    label = decode_predictions(out)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))