"""
Class for managing our data.
"""
import csv
import math
from collections import defaultdict

import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from keras.utils import to_categorical
from tqdm import tqdm

from solution1.objects.Singleton import Singleton
from solution3.config import Config
from solution3.processor import process_image


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


class Dataset(metaclass=Singleton):

    def __init__(self, seq_length=40):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.sequence_path = os.path.join(Config.root_data, 'sequences', str(seq_length))
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.classes, self.data = self.get_data()

    def get_data(self):
        classes = list()

        class_dirs = os.listdir(Config.root_img_seq_dataset)
        random.shuffle(class_dirs) if Config.shuffle_classes else ...

        # Choose classes
        if Config.class_list:
            for class_name in Config.class_list:
                if class_name in class_dirs:
                    classes.append(class_name)
                else:
                    print(f"{class_name} class does not exist in dataset. Skipping.")
        else:
            for class_name in class_dirs:
                if len(classes) == Config.class_number:
                    break
                classes.append(class_name)

        # Choose videos for a class
        items = []  # tuple(label, path)
        for class_name in tqdm(classes):
            video_folders = glob.glob(f"{Config.root_img_seq_dataset}/{class_name}/*")
            random.shuffle(video_folders) if Config.shuffle_videos else ...

            temp_items = []
            for video_path in video_folders:
                if len(temp_items) == Config.video_number_per_class:
                    break
                elif not os.path.exists(os.path.join(video_path, str(self.seq_length), Config.npy_filename)):
                    continue
                temp_items.append((class_name, video_path))

            items.extend(temp_items)
        return classes, items

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []

        test_number = math.ceil(len(list(self.data)) * Config.test_split)
        temp_data = self.data.copy()
        random.shuffle(temp_data)
        train.extend(temp_data[test_number:])
        test.extend(temp_data[:test_number])

        return train, test

    def get_all_sequences_in_memory(self):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()

        print("Loading %d samples into memory for %sing." % (len(train), "train"))
        print("Loading %d samples into memory for %sing." % (len(test), "test"))

        X_train, y_train, X_test, y_test = [], [], [], []
        for row in train:
            sequence = self.get_extracted_sequence(row[1])
            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise
            X_train.append(sequence)
            y_train.append(self.get_class_one_hot(row[0]))

        for row in test:
            sequence = self.get_extracted_sequence(row[1])
            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise
            X_test.append(sequence)
            y_test.append(self.get_class_one_hot(row[0]))

        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    @threadsafe_generator
    def train_frame_generator(self, batch_size):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()

        print("Creating %s generator with %d samples." % ("train", len(train)))
        for item in train:
            print(item, end="\n")

        while 1:
            X_train, y_train = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Get a random sample.
                sample = random.choice(train)

                # Get the sequence from disk.
                sequence = self.get_extracted_sequence(sample[1])
                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                X_train.append(sequence)
                y_train.append(self.get_class_one_hot(sample[0]))

            yield np.array(X_train), np.array(y_train)

    @threadsafe_generator
    def test_frame_generator(self, batch_size):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()

        print("Creating %s generator with %d samples." % ("train", len(train)))
        for item in test:
            print(item)


        while 1:
            X_test, y_test = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Get a random sample.
                sample = random.choice(test)

                # Get the sequence from disk.
                sequence = self.get_extracted_sequence(sample[1])
                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                X_test.append(sequence)
                y_test.append(self.get_class_one_hot(sample[0]))

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
        return os.path.join(video_path, str(self.seq_length), Config.npy_filename)

    def get_frames_by_filename(self, video_dir):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[1] == video_dir:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % video_dir)

        # Get the sequence from disk.
        sequence = self.get_extracted_sequence(sample[1])

        if sequence is None:
            raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

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
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))


if __name__ == '__main__':
    d = Dataset()
    d.split_train_test()
    d.get_class_one_hot('ApplyLipstick')
