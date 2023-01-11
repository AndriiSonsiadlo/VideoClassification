import logging
import os
import cv2
import random
import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from definitions import ROOT_DIR
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from src.utils.utils import get_shortest_list

logger = logging.getLogger(__name__)

seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

#data visualization
# plt.figure(figsize=(30,30))
# ignored = ['.DS_Store']
# all_classes_names = [x for x in os.listdir(CLASS_PATH) if x not in ignored]
#
# # Generate a random sample of images each time the cell runs
# random_range = random.sample(range(len(all_classes_names)), 3)

# for counter, random_index in enumerate(random_range, 1):
#     selected_class_name = all_classes_names[random_index]
#     video_files_name_list = os.listdir(CLASS_PATH + f'{selected_class_name}')
#     selected_video_file = random.choice(video_files_name_list)
#     if selected_video_file.startswith("."):
#         continue
#     # first_file = os.listdir(CLASS_PATH + f'{selected_class_name}/{selected_video_file}')[0]
#     video_reader = cv2.VideoCapture(CLASS_PATH + f'{selected_class_name}/{selected_video_file}')
#     _, bgr_frame = video_reader.read()
#     video_reader.release()
#     rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
#     cv2.putText(rgb_frame, selected_class_name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     plt.subplot(3, 1, counter)
#     plt.imshow(rgb_frame)
#     plt.axis('off')



#STEP 3
# classes_list = ["Basketball", "VolleyballSpiking", "PushUps", "ShavingBeard", "TableTennisShot", "TennisSwing", "PullUps", "Punch", "Skiing", "Surfing"]

def process_frames(video_path, img_size, normalize=True):
    frames_list = []
    capture = cv2.VideoCapture(video_path)

    while capture.isOpened():
        i = 0
        while i!=3:
            success, frame = capture.read()
            i += 1
            if not success:
                break
            frame = resize(frame, img_size)
            if normalize:
                frame = frame/255
            frames_list.append(frame)

    capture.release()
    return frames_list

def create_dataset(src_dir, class_list):
    temp_videos_list = []
    videos_list = []
    labels = []
    class_names = []

    videos_dir = os.path.join(ROOT_DIR, src_dir)

    logger.info(f'Begin Processing video frames')
    for class_index, class_name in enumerate(class_list):
        logger.info(f'Extracting data for: {class_name}')
        files_list = os.listdir(os.path.join(videos_dir, class_name))
        for file in files_list:
            video_file_path = os.path.join(videos_dir, class_name, file)
            frames = process_frames(video_file_path, (64, 64, 3))
            temp_videos_list.append(frames)

        logger.debug(f'{class_name} temp_videos len({len(temp_videos_list)})')
        videos_list.extend(temp_videos_list)
        class_names.extend([class_name] * len(temp_videos_list))
        labels.extend([class_index] * len(temp_videos_list))
        temp_videos_list.clear()

    # WEZ NAJMNIEJSZA LISTE, pamietaj o skroceniu tez
    shortest_video_list_len = get_shortest_list(videos_list)
    logger.debug(f'shortest: {shortest_video_list_len}')
    short_videos_list = [video[:shortest_video_list_len] for video in videos_list]

    videos_list = np.array(short_videos_list)

    print(len(short_videos_list[0]))
    # print(videos_list)

    labels = np.array(labels)
    class_names = np.array(class_names)
    logger.info('Finished creating dataset')
    return videos_list, labels, class_names

# with open(r'labels/lb.pickle', 'wb') as f:
#     pickle.dump(lb, f)

def spatial_cnn(input_shape, classes):
    model_output_size = len(classes)

    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(model_output_size, activation= 'softmax'))

    print(model.summary())
    logger.debug("Model Created Successfully!")

    return model

def model_fit(model, videos_list, encoded_labels):
    features_train, features_test, labels_train, labels_test = train_test_split(videos_list, encoded_labels,
                                                                                test_size=0.3, shuffle=True,
                                                                                random_state=seed_constant)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    logger.info('Compiling model')
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

    logger.info('Training Model')
    model_training_history = model.fit(x=features_train,
                                       y=labels_train,
                                       epochs=5,
                                       batch_size=4,
                                       shuffle=True,
                                       validation_split=0.2,
                                       callbacks=[early_stopping_callback])

    model_evaluation_history = model.evaluate(features_test, labels_test)

    # saving model weights
    logger.info('Saving Model')
    model_save(model, model_evaluation_history)

    return model_training_history, model_evaluation_history

def model_save(model, model_evaluation_history):
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    model_name = f'Model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

    # saving model
    dst_dir = os.path.join(ROOT_DIR, 'models/', model_name)
    model.save(dst_dir)
    logger.info('model saved')

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
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

def visualize_model_training(model_training_history):
    acc = model_training_history.history['accuracy']
    val_acc = model_training_history.history['val_accuracy']

    loss = model_training_history.history['loss']
    val_loss = model_training_history.history['val_loss']

    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_alpha(0.5)

    plt.subplot(2, 1, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.plot([100 - 1, 100 - 1], plt.ylim())
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Start Fine Tuning'],
               loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.plot([100 - 1, 100 - 1], plt.ylim())
    plt.legend(['Training Loss', 'Validation Loss', 'Start Fine Tuning'],
               loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epochs')
    plt.title('Loss')
    plt.savefig(r'D:\MAGISTERSKIE\Uczenie_glebokie\Action_Classification_CNN\videoclassification\figures' + "/training_plot1.pdf")
    plt.show()

if __name__ == '__main__':
    input_shape = (64, 64, 3)
    classes_list = ["Basketball", "PushUps"]
    video_src = 'data/videos'
    videos_list, labels, class_names = create_dataset(video_src, classes_list)
    one_hot_encoded_labels = to_categorical(labels)

    model = spatial_cnn(input_shape, classes_list)
    # plot_model(model, to_file = 'model_structure_plot.png', show_shapes = True, show_layer_names = True)
    lb = classes_list
    train_history, eval_history = model_fit(model, videos_list, one_hot_encoded_labels)

    plot_metric(train_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    plot_metric(train_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')
