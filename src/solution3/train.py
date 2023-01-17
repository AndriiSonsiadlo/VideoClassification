"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import Models
from Dataset import Dataset
import time
import os.path

from solution3.config import Config


def train(data_type, seq_length, model, saved_model=None,
          class_limit=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
                              '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
                                        str(timestamp) + '.log'))

    # Get the data and process it.
    data = Dataset(
        seq_length=seq_length,
        class_limit=class_limit
    )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = Models(len(data.classes), model, seq_length, saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger], #, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)

    rm.model.save(f"model-{model}")


def main():
    """These are the main training settings. Set each before running
    this file."""
    model = Config.model
    saved_model = Config.saved_model
    class_limit = Config.class_limit_lrn
    seq_length = Config.seq_length_lrn
    load_to_memory = Config.load_to_memory  # preload the sequences into memory
    batch_size = Config.batch_size
    nb_epoch = Config.nb_epoch

    data_type = 'features'

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)


if __name__ == '__main__':
    main()