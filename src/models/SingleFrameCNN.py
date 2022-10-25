import logging
import os.path

import tensorflow as tf
keras = tf.keras
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
import datetime as dt
from src.models.BaseModel import BaseModel





class SingleFrameCNN(BaseModel):

    def __init__(self, name="basic_SF_CNN", seed=23, split=0.8):
        super().__init__(name)
        self.seed = 23
        self.split = 0.8
        self.model = None

        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __str__(self):
        return f'<{self.__name__}>{self.name}'

    def create_model(self, model_output_size, image_height=240, image_width=320):
        model = Sequential()

        # Defining The Model Architecture
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(model_output_size, activation='softmax'))

        # Printing the models summary
        model.summary()

        return model

