"""
Class for managing our data.
"""
import glob
import math
import operator
import os.path
import random
import threading

import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm

from src.solution3.config import Config
from src.solution3.objects.ActionClass import ActionClass
from src.solution3.objects.ActionElement import ActionElement
from src.solution3.processor import process_image


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""

    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))

    return gen


class Dataset():
    cfg = Config()

    def __init__(self, seq_length=40, class_number=10, shuffle_classes=False, video_number_per_class=10,
                 shuffle_videos=False, incl_classes: tuple = (), test_split=0.3, class_list: list | None = None):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.action_classes = self.get_data(class_number, incl_classes, shuffle_classes, video_number_per_class,
                                            shuffle_videos, class_list, test_split)

    def get_classes(self):
        classes = list(self.action_classes.keys())
        return classes

    def get_data(self, class_number, incl_classes, shuffle_classes, video_number_per_class, shuffle_videos, class_list,
                 test_split):
        action_classes = dict()

        class_dirs = os.listdir(self.cfg.root_img_seq_dataset)
        random.shuffle(class_dirs) if shuffle_classes else ...

        # Choose classes
        classes = []
        for incl_class in incl_classes:
            if incl_class in class_dirs:
                classes.append(incl_class)

        for action in class_dirs:
            if len(classes) >= class_number:
                break
            if action not in classes:
                classes.append(action)

        classes.sort()

        # Choose videos for a class
        for class_name in tqdm(classes):
            video_folders = glob.glob(f"{self.cfg.root_img_seq_dataset}/{class_name}/*")
            random.shuffle(video_folders) if shuffle_videos else ...

            videos_per_class = []
            for video_path in video_folders:
                action_element = ActionElement(video_path, class_name)
                if len(videos_per_class) == video_number_per_class:
                    break
                elif not action_element.exists_npy(str(self.seq_length)):
                    print(f".npy does not exists: {action_element.video_folder_path}")
                    continue
                videos_per_class.append(action_element)

            train, test = self.split_train_test(videos_per_class, test_split)
            action_class = ActionClass(class_name=class_name, train_list=train, test_list=test, test_split=test_split)
            action_classes[class_name] = action_class
        return action_classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        classes = self.get_classes()

        label_encoded = classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.action_classes))

        assert len(label_hot) == len(classes)

        return label_hot

    def split_train_test(self, videos, test_split):
        """Split the data into train and test groups."""
        train = []
        test = []

        test_number = math.ceil(len(videos) * test_split)
        random.shuffle(videos)
        train.extend(videos[test_number:])
        test.extend(videos[:test_number])

        return train, test

    def get_train_test_lists(self):
        train = []
        test = []

        for class_name, action_class in self.action_classes.items():
            train.extend(action_class.train_list)
            test.extend(action_class.test_list)

        return train, test

    def get_all_sequences_in_memory(self):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.get_train_test_lists()

        print("Loading %d samples into memory for %sing." % (len(train), "train"))
        print("Loading %d samples into memory for %sing." % (len(test), "test"))

        X_train, y_train, X_test, y_test = [], [], [], []
        for act in train:
            sequence = self.get_extracted_sequence(act.video_folder_path)
            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise
            X_train.append(sequence)
            y_train.append(self.get_class_one_hot(act.class_name))

        for act in test:
            sequence = self.get_extracted_sequence(act.video_folder_path)
            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise
            X_test.append(sequence)
            y_test.append(self.get_class_one_hot(act.class_name))

        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    @threadsafe_generator
    def train_frame_generator(self, batch_size):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.get_train_test_lists()

        print("Creating %s generator with %d samples." % ("train", len(train)))
        for item in train:
            print(item, end="\n")

        while 1:
            X_train, y_train = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Get a random sample.
                act = random.choice(train)

                # Get the sequence from disk.
                sequence = self.get_extracted_sequence(act.video_folder_path)
                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                X_train.append(sequence)
                y_train.append(self.get_class_one_hot(act.class_name))

            yield np.array(X_train), np.array(y_train)

    @threadsafe_generator
    def test_frame_generator(self, batch_size):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.get_train_test_lists()

        print("Creating %s generator with %d samples." % ("test", len(test)))
        for item in test:
            print(item)

        while 1:
            X_test, y_test = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Get a random sample.
                act = random.choice(test)

                # Get the sequence from disk.
                sequence = self.get_extracted_sequence(act.video_folder_path)
                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                X_test.append(sequence)
                y_test.append(self.get_class_one_hot(act.class_name))

            yield np.array(X_test), np.array(y_test)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x) for x in frames]

    def get_extracted_sequence(self, video_path):
        """Get the saved extracted features."""

        path = self.generate_npy_path(video_path)
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def generate_npy_path(self, video_path):
        return os.path.join(video_path, str(self.seq_length), self.cfg.npy_filename)

    @staticmethod
    def get_frames_for_sample(video_dir):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        images = sorted(glob.glob(os.path.join(video_dir, '*jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        if len(input_list) <= size and len(input_list) > 0:
            return input_list

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.get_classes()):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            # if i > nb_to_return - 1 or class_prediction[1] == 0.0:
            #     break
            print("%s: %.4f" % (class_prediction[0], class_prediction[1]))

        return sorted_lps
