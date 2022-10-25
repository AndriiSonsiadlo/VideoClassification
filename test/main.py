from matplotlib import pyplot as plt

from src.models.SingleFrameCNN import *
from src.utils.data_prep import *
from src.utils.data_analysis import predict_on_video, make_average_predictions
import logging
logger = logging.getLogger('test_main')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


def test_CNN_class():
    seed_constant = 40

    logging.debug(f'CREATING DATASET')
    actions_cls = ['Basketball', 'VolleyballSpiking']
    logger.debug(f'creating dataset for classes: {actions_cls}')

    frames_per_video = 3
    extract_process(action_dirs=actions_cls, frames_per_video=frames_per_video)
    features, labels = create_dataset(action_classes=actions_cls, frames_per_video=frames_per_video)

    sf_cnn = SingleFrameCNN(actions_cls, seed=seed_constant, name="Fox_CNN")

    sf_cnn.create_model()
    sf_cnn.train(features, labels, epochs=10, batch_size=3)
    sf_cnn.plot_metric('loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    sf_cnn.plot_metric('accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

    # Setting the Window Size which will be used by the Rolling Average Process
    window_size = 25

    test_video_names = list()
    for action in actions_cls:
        input_action_file_path = os.path.join(VIDEOS_BASE_PATH, action)
        input_video_file_name = random.sample(os.listdir(input_action_file_path), 1)
        test_video_names.append(f'/{action}/{input_video_file_name}')

    # grab a random video from 1 of the action classes
    input_video_file_path = os.path.join(VIDEOS_BASE_PATH, test_video_names[random.randint(0, len(actions_cls) - 1)])
    # Calling the predict_on_live_video method to start the Prediction and Rolling Average Process
    predict_on_video(input_video_file_path, window_size)

def test():
    seed_constant = 23

    logging.debug(f'CREATING DATASET')
    actions_cls = ['Basketball', 'VolleyballSpiking']
    logger.debug(f'creating dataset for classes: {actions_cls}')

    frames_per_video = 3
    extract_process(action_dirs=actions_cls, frames_per_video=frames_per_video)
    features, labels = create_dataset(action_classes=actions_cls, frames_per_video=frames_per_video)


    one_hot_encoded_labels = to_categorical(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                                test_size=0.2, shuffle=True,
                                                                                random_state=seed_constant)
    sf_cnn = SingleFrameCNN(seed=seed_constant)

    model = sf_cnn.create_model(len(actions_cls))

    # Adding Early Stopping Callback
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

    # Adding loss, optimizer and metrics values to the model.
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

    # Start Training
    logger.debug(f'start training of model')
    logger.debug(f'num train features: {len(features_train)}')
    model_training_history = model.fit(x=features_train, y=labels_train, epochs=10, batch_size=4, shuffle=True,
                                       validation_split=0.2, callbacks=[early_stopping_callback])

    model_evaluation_history = model.evaluate(features_test, labels_test)

    logger.debug('plotting model training results')
    # plot_metric(model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    # plot_metric(model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

    # Setting the Window Size which will be used by the Rolling Average Process
    window_size = 25

    test_video_names = list()
    for action in actions_cls:
        input_action_file_path = os.path.join(VIDEOS_BASE_PATH, action)
        input_video_file_name = random.sample(os.listdir(input_action_file_path), 1)
        test_video_names.append(f'/{action}/{input_video_file_name}')

    # grab a random video from 1 of the action classes
    input_video_file_path = os.path.join(VIDEOS_BASE_PATH, test_video_names[random.randint(0, len(actions_cls)-1)])
    # Calling the predict_on_live_video method to start the Prediction and Rolling Average Process
    predict_on_video(model, input_video_file_path, window_size, actions_cls)

    # make_average_predictions(model, input_video_file_path, 50, actions_cls)


if __name__ == '__main__':
    test_CNN_class()