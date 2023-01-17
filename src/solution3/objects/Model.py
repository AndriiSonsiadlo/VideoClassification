import os
import time
import uuid

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from solution3.objects.Dataset import Dataset
from solution3.objects.ModelLoader import ModelLoader
from solution3.config import Config


class Model:
    cfg = Config()

    def __init__(self, model_name="lstm", batch_size=32, nb_epoch=100, class_number=10, shuffle_classes=False,
                 video_number_per_class=10, shuffle_videos=False,
                 seq_length=40, load_to_memory=False, save_path: str = None, test_split=0.3):

        self.data = Dataset(seq_length=seq_length, class_number=class_number, shuffle_classes=shuffle_classes,
                            video_number_per_class=video_number_per_class, shuffle_videos=shuffle_videos,
                            test_split=test_split)

        self.model_name = model_name
        self.model = ModelLoader(model_name=model_name, nb_classes=len(self.data.action_classes.keys()),
                                 seq_length=seq_length, saved_model=None)

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.test_split = test_split

        self.load_to_memory = load_to_memory

        if save_path is None:
            id = uuid.uuid4()
            self.save_path = os.path.join(self.cfg.root_models, str(id))
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        else:
            if not os.path.exists(save_path):
                self.save_path = save_path
                os.makedirs(save_path)

    def train(self):
        # Helper: Save the model.
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(self.save_path, 'checkpoints', self.model_name + '-' + "features" + \
                                  '.{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True)

        # Helper: TensorBoard
        tb = TensorBoard(log_dir=os.path.join(self.save_path, 'logs', self.model_name))

        # Helper: Stop when we stop learning.
        early_stopper = EarlyStopping(patience=5)

        # Helper: Save results.
        timestamp = time.time()
        csv_logger = CSVLogger(os.path.join(self.save_path, 'logs', self.model_name + '-' + 'training-' + \
                                            str(timestamp) + '.log'))

        # Get samples per epoch.
        # Multiply by $test_split$ to attempt to guess how much of data.data is the train set.

        train, test = self.data.get_train_test_lists()
        steps_per_epoch = (len([*train, *test]) * (1 - self.test_split)) // self.batch_size

        if self.load_to_memory:
            # Get data.
            X, y, X_test, y_test = self.data.get_all_sequences_in_memory()
        else:
            # Get generators.
            generator = self.data.train_frame_generator(self.batch_size)
            val_generator = self.data.test_frame_generator(self.batch_size)

        # Fit!
        if self.load_to_memory:
            # Use standard fit.
            self.model.model.fit(
                X,
                y,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger],
                epochs=self.nb_epoch)
            self.model.model.save('my_model.h5')
        else:
            # Use fit generator.
            self.model.model.fit_generator(
                generator=generator,
                steps_per_epoch=steps_per_epoch,
                epochs=self.nb_epoch,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger, checkpointer],
                validation_data=val_generator,
                validation_steps=40,
                workers=4)
