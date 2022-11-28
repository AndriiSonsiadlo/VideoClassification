import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
#from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline
 
from sklearn.model_selection import train_test_split
 
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model




CLASS_PATH = 'data/videos/'

seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

#data visualization
plt.figure(figsize=(30,30))
ignored = ['.DS_Store']
all_classes_names = [x for x in os.listdir(CLASS_PATH) if x not in ignored]

# Generate a random sample of images each time the cell runs
random_range = random.sample(range(len(all_classes_names)), 3)

for counter, random_index in enumerate(random_range, 1):
    selected_class_name = all_classes_names[random_index]
    video_files_name_list = os.listdir(CLASS_PATH + f'{selected_class_name}')
    selected_video_file = random.choice(video_files_name_list)
    # first_file = os.listdir(CLASS_PATH + f'{selected_class_name}/{selected_video_file}')[0]
    video_reader = cv2.VideoCapture(CLASS_PATH + f'{selected_class_name}/{selected_video_file}')
    _, bgr_frame = video_reader.read()
    video_reader.release()
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    cv2.putText(rgb_frame, selected_class_name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    plt.subplot(3, 1, counter)
    plt.imshow(rgb_frame)
    plt.axis('off')



#STEP 3
img_height, img_width = 64, 64
max_class_img = 8000
classes_list = ["Basketball", "VolleyballSpiking", "PushUps"]

model_output_size = len(classes_list)

def extracion_frames(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)

    while True:
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (img_height, img_width))
        normalized_frame = resized_frame/255
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list


def create_dataset():
    temp_features = []
    features = []
    labels = []

    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting data for: {class_name}')
        files_list = os.listdir(os.path.join(CLASS_PATH, class_name))
        for file in files_list:
            video_file_path = os.path.join(CLASS_PATH, class_name, file)
            frames = extracion_frames(video_file_path)
            temp_features.extend(frames)

    features.extend(random.sample(temp_features, max_class_img))
    labels.extend([class_index] * max_class_img)
    temp_features.clear()

    np.asarray(features)
    np.array(labels)

    return features, labels

features, labels = create_dataset()

one_hot_encoded_labels = to_categorical(labels)

features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.2, shuffle=True, random_state = seed_constant)


def create_model():
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = (img_height, img_width, 3)))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(model_output_size, activation= 'softmax'))


    model.summary()
 
    return model
 
 

model = create_model()
 
print("Model Created Successfully!")

# plot_model(model, to_file = 'model_structure_plot.png', show_shapes = True, show_layer_names = True)


early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

model_training_history = model.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4 , shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])




model_evaluation_history = model.evaluate(features_test, labels_test)

date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
model_name = f'Model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'
 
# Saving your Model
model.save(model_name)



def plot_metric(metric_name_1, metric_name_2, plot_name):
  # Get Metric values using metric names as identifiers
  metric_value_1 = model_training_history.history[metric_name_1]
  metric_value_2 = model_training_history.history[metric_name_2]
 
  # Constructing a range object which will be used as time 
  epochs = range(len(metric_value_1))
   
  # Plotting the Graph
  plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
  plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
   
  # Adding title to the plot
  plt.title(str(plot_name))
 
  # Adding legend to the plot
  plt.legend()

  plot_metric('loss', 'val_loss', 'Total Loss vs Total Validation Loss')
  plot_metric('accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

