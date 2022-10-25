import logging
import os.path
from collections import deque
import cv2 as cv
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

    def __init__(self, classes: list[str], name="basic_SF_CNN", seed=23, validation_split=0.2):
        super().__init__(name)
        self.image_h = None
        self.image_w = None
        self.seed = seed
        self.v_split = validation_split
        self.cls_list = classes
        self.model = None
        self.output_model_size = len(self.cls_list)
        self.model_train_hist = None
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __str__(self):
        return f'<{self.__name__}>{self.name}'

    def create_model(self, image_height=240, image_width=320):
        model = Sequential()
        self.image_h = image_height
        self.image_w = image_width
        # Defining The Model Architecture
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.output_model_size, activation='softmax'))

        self.model = model
        # Printing the models summary
        model.summary()

        return model

    def train(self, features, labels, test_size=0.2, epochs=50, batch_size=5) -> None:
        one_hot_encoded_labels = to_categorical(labels)
        features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                                    test_size=test_size, shuffle=True,
                                                                                    random_state=self.seed)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
        # Adding Early Stopping Callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

        # Adding loss, optimizer and metrics values to the model.
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

        self.model_train_hist = self.model.fit(x=features_train, y=labels_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                           validation_split=self.v_split, callbacks=[early_stopping_callback])

        return self.model_train_hist

    def plot_metric(self, metric_name_1, metric_name_2, plot_name):
        if self.model_train_hist is not None:
            # Get Metric values using metric names as identifiers
            metric_value_1 = self.model_train_hist.history[metric_name_1]
            metric_value_2 = self.model_train_hist.history[metric_name_2]

            # Constructing a range object which will be used as time
            epochs = range(len(metric_value_1))

            # Plotting the Graph
            plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
            plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

            # Adding title to the plot
            plt.title(str(plot_name))

            # Adding legend to the plot
            plt.legend()

    def predict_on_video(self, video_file_path, window_size):
        logging.debug(f'making prediction on video')
        # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
        predicted_labels_probabilities_deque = deque(maxlen=window_size)
        # Reading the Video File using the VideoCapture Object
        video_reader = cv.VideoCapture(video_file_path)

        # Getting the width and height of the video
        original_video_width = int(video_reader.get(cv.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv.CAP_PROP_FRAME_HEIGHT))

        while True:
            # Reading The Frame
            success, frame = video_reader.read()
            if not success:
                logging.debug(f'PREDICT_VIDEO(): frame reading failed')
                break
            # Resize the Frame to fixed Dimensions
            resized_frame = cv.resize(frame, (self.image_h,
                                              self.image_w)) if self.image_h != original_video_height or self.image_w != original_video_width else frame

            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
            normalized_frame = resized_frame / 255

            # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
            predicted_labels_probabilities = self.model.predict(np.expand_dims(normalized_frame, axis=0))[0]

            # Appending predicted label probabilities to the deque object
            predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

            # Assuring that the Deque is completely filled before starting the averaging process
            if len(predicted_labels_probabilities_deque) == window_size:
                # Converting Predicted Labels Probabilities Deque into Numpy array
                predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

                # Calculating Average of Predicted Labels Probabilities Column Wise
                predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis=0)

                # Converting the predicted probabilities into labels by returning the index of the maximum value.
                predicted_label = np.argmax(predicted_labels_probabilities_averaged)

                # Accessing The Class Name using predicted label.
                predicted_class_name = self.cls_list[predicted_label]

                # Overlaying Class Name Text Ontop of the Frame
                cv.putText(frame, predicted_class_name, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # cv.destroyAllWindows()

        # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
        video_reader.release()