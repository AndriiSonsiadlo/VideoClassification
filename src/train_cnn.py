import logging
import tensorflow as tf
import os
import keras.callbacks
from models.temporal_CNN import temporal_cnn
from keras.optimizers import SGD
from definitions import ROOT_DIR
from src.utils.optical_flow import sequence_generator
logger = logging.getLogger(__name__)

BatchSize = 16
def fit_model(model, train_data, test_data, weights_dir, input_shape, n_classes, optical_flow=True):
    try:
        train_generator = sequence_generator(train_data, BatchSize, input_shape, n_classes)
        test_generator = sequence_generator(test_data, BatchSize, input_shape, n_classes)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        print('Start fitting model')
        while True:
            checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
            earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=2, mode='auto')
            tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(ROOT_DIR, 'logs'), histogram_freq=0, write_graph=True, write_images=True)
            model.fit_generator(
                train_generator,
                steps_per_epoch=200,
                epochs=30,
                validation_data=test_generator,
                validation_steps=100,
                verbose=2,
                callbacks=[checkpointer, tensorboard, earlystopping]
            )

    except KeyboardInterrupt:
        print('Training is interrupted')