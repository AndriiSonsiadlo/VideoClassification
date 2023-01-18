import json
import os
import pickle
import time
import uuid

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from matplotlib import pyplot as plt

from src.solution3.config import Config
from src.solution3.objects.Dataset import Dataset
from src.solution3.objects.ModelLoader import ModelLoader


class ModelData:
    cfg = Config()

    def __init__(self, model_name="lstm", batch_size=32, nb_epoch=100, class_number=10, shuffle_classes=False,
                 video_number_per_class=10, shuffle_videos=False, incl_classes=(),
                 seq_length=40, load_to_memory=False, save_path: str = None, test_split=0.3):

        self.data = Dataset(seq_length=seq_length, class_number=class_number, shuffle_classes=shuffle_classes,
                            video_number_per_class=video_number_per_class, shuffle_videos=shuffle_videos,
                            test_split=test_split, incl_classes=incl_classes)

        self.model_name = model_name
        self.model = ModelLoader(model_name=model_name, nb_classes=len(self.data.get_classes()),
                                 seq_length=seq_length, saved_model=None)

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.test_split = test_split
        self.class_number = class_number
        self.video_number_per_class = video_number_per_class

        self.load_to_memory = load_to_memory

        if save_path is None:
            id = uuid.uuid4()
            self.save_path = os.path.join(self.cfg.root_models,
                                          f"{self.model_name}-{class_number}classes-{video_number_per_class}videos-{str(id)}")
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

        # Fit!
        if self.load_to_memory:
            # Get data.
            X, y, X_test, y_test = self.data.get_all_sequences_in_memory()
            # Use standard fit.
            history = self.model.model.fit(
                X,
                y,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger],
                epochs=self.nb_epoch)
        else:
            # Get generators.
            generator = self.data.train_frame_generator(self.batch_size)
            val_generator = self.data.test_frame_generator(self.batch_size)
            time.sleep(3)
            # Use fit generator.
            history = self.model.model.fit_generator(
                generator=generator,
                steps_per_epoch=steps_per_epoch,
                epochs=self.nb_epoch,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger],
                validation_data=val_generator,
                validation_steps=40,
                workers=4)
        self.model.model.save(os.path.join(self.save_path, 'model'))
        self.model = None

        return history

    def train_plot(self, history, start_ft=0):
        acc = history['accuracy']
        val_acc = history['val_accuracy']

        loss = history['loss']
        val_loss = history['val_loss']

        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_alpha(0.5)

        plt.subplot(2, 1, 1)
        plt.plot(acc)
        plt.plot(val_acc)

        legend = ['Training Accuracy', 'Validation Accuracy']
        if start_ft != 0:
            plt.plot([start_ft - 1, start_ft - 1], plt.ylim())
            legend.append('Start Fine Tuning')

        plt.legend(legend, loc='lower right')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss)
        plt.plot(val_loss)

        legend = ['Training Loss', 'Validation Loss']
        if start_ft != 0:
            plt.plot([start_ft - 1, start_ft - 1], plt.ylim())
            legend.append('Start Fine Tuning')

        plt.legend(legend, loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.xlabel('Epochs')
        plt.title('Loss')
        plt.savefig(os.path.join(self.save_path, "plot.png"))
        plt.show()

    def save_to_json(self):
        train, test = self.data.get_train_test_lists()
        train = [train_el.video_folder_path for train_el in train]
        test = [test_el.video_folder_path for test_el in test]

        json_data = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "nb_epoch": self.nb_epoch,
            "seq_length": self.seq_length,
            "class_number": len(self.data.get_classes()),
            "classes": self.data.get_classes(),
            "video_number_per_class": self.video_number_per_class,
            "test_split": self.test_split,
            "train_list": train,
            "test_list": test,
        }

        # Serializing json
        json_object = json.dumps(json_data, indent=4)

        # Writing to sample.json
        json_path = os.path.join(self.save_path, "data.json")
        with open(json_path, "w") as outfile:
            outfile.write(json_object)
            print(f"Model data was saved in json: {json_path}")


def save_pickle_model(model: ModelData):
    try:
        with open(os.path.join(model.save_path, "data.pickle"), 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
            print(fr"Model data was saved in pickle: {model.save_path}\data.pickle")
    except Exception:
        raise (f"Cannot save .pickle {model.save_path}")


def load_pickle_model(path_to_model: str):
    with open(path_to_model, "rb") as input_file:
        model = pickle.load(input_file)
        print(f"Model was loaded: {model.save_path}")
    return model
