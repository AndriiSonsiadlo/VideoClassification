import os
import logging
import sys

# instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# define handler and formatter
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
# add formatter to handler
handler.setFormatter(formatter)
# add handler to logger
logger.addHandler(handler)

import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D
from keras.applications import ResNet50
from keras.layers import Dropout, Flatten, Dense, Input, Convolution2D, MaxPooling2D, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def temporal_cnn(input_shape, classes, weights_dir, include_top=True):
    optical_flow_input = Input(shape=input_shape)

    x = Convolution2D(96, kernel_size=(7, 7), strides=(2, 2), padding='same', name='tmp_conv1')(optical_flow_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', name='tmp_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv4')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='tmp_conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='tmp_fc6')(x)
    x = Dropout(0.9)(x)

    x = Dense(2048, activation='relu', name='tmp_fc7')(x)
    x = Dropout(0.9)(x)

    if include_top:
        x = Dense(classes, activation='softmax', name='tmp_fc101')(x)

    model = Model(inputs=optical_flow_input, outputs=x, name='temporal_CNN')

    if os.path.exists(weights_dir):
        model.load_weights(weights_dir, by_name=True)

    return model


if __name__ == '__main__':
    input_shape = (216, 216, 18)
    N_CLASSES = 101
    model = temporal_cnn(input_shape, N_CLASSES, weights_dir='')
    print(model.summary())

    model = temporal_cnn()