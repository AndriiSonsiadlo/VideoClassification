"""
Given a video path and a saved model (checkpoint), produce classification
predictions.

Note that if using a model that requires features to be extracted, those
features must be extracted first.

Note also that this is a rushed demo script to help a few people who have
requested it and so is quite "rough". :)
"""
import os
import uuid

from PIL.Image import Image
from keras.models import load_model
from Dataset import Dataset
import numpy as np

import cv2

from config import Config
from solution3.extractor import Extractor


def get_frames_from_video(path: str):
    capture = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            # BGR to RGB
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) > 300:
                break
    finally:
        capture.release()
    return np.array(frames)


def predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit):
    model = load_model(saved_model)

    # Get the data and process it.
    if image_shape is None:
        data = Dataset(seq_length=seq_length, class_limit=class_limit)
    else:
        data = Dataset(seq_length=seq_length, image_shape=image_shape,
                       class_limit=class_limit)

    # Extract the sample from the data.
    sample = data.get_frames_by_filename(video_name, data_type)

    # Predict!
    prediction = model.predict(np.expand_dims(sample, axis=0))
    print(prediction)
    data.print_class_from_prediction(np.squeeze(prediction, axis=0))


def predict2(sequence, saved_model, image_shape, class_limit):
    model = load_model(saved_model)

    # Get the data and process it.
    if image_shape is None:
        data = Dataset(seq_length=40, class_limit=class_limit)
    else:
        data = Dataset(seq_length=40, image_shape=image_shape,
                       class_limit=class_limit)

    # Predict!
    prediction = model.predict(np.expand_dims(sequence, axis=0))
    print(prediction)
    data.print_class_from_prediction(np.squeeze(prediction, axis=0))


def predict_from_npy(npy_path, saved_model, class_limit):
    model = load_model(saved_model)

    # Get the data and process it.
    data = Dataset(seq_length=40, class_limit=class_limit)

    sample = np.load(npy_path)

    # Predict!
    prediction = model.predict(np.expand_dims(sample, axis=0))
    print(prediction)
    data.print_class_from_prediction(np.squeeze(prediction, axis=0))


def main():
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    model = 'lstm'
    # Must be a weights file.
    saved_model = r'C:\VMShare\videoclassification\data\checkpoints\lstm-features.001-1.365.hdf5'
    # Sequence length must match the length used during training.
    seq_length = 40
    # Limit must match that used during training.
    class_limit = 5

    # Demo file. Must already be extracted & features generated (if model requires)
    # Do not include the extension.
    # Assumes it's in data/[train|test]/
    # It also must be part of the train/test data.
    # TODO Make this way more useful. It should take in the path to
    # an actual video file, extract frames, generate sequences, etc.
    # video_name = 'v_Archery_g04_c02'
    video_name = "v_Archery_g02_c02"

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit)


def main2(feature_sequence):
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    model = 'lstm'
    # Must be a weights file.
    saved_model = r'C:\VMShare\videoclassification\data\checkpoints\lstm-features.034-0.144.hdf5'
    # Sequence length must match the length used during training.
    seq_length = 40
    # Limit must match that used during training.
    class_limit = 5

    # Demo file. Must already be extracted & features generated (if model requires)
    # Do not include the extension.
    # Assumes it's in data/[train|test]/
    # It also must be part of the train/test data.
    # TODO Make this way more useful. It should take in the path to
    # an actual video file, extract frames, generate sequences, etc.
    # video_name = 'v_Archery_g04_c02'

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    predict2(feature_sequence, saved_model, image_shape, class_limit)


if __name__ == '__main__':
    # main()


    # from skimage.transform import resize
    #
    # model = Extractor()
    # frames = get_frames_from_video(r"C:\VMShare\videoclassification\data\train\Bowling\v_Bowling_g02_c04\v_Bowling_g02_c04.avi")
    # frames = Dataset.rescale_list(frames, 40)
    # sequence = []
    # for image in frames:
    #     image = resize(image, (299, 299))
    #     features = model.extract_from_frame(image)
    #     sequence.append(features)
    # # main2(sequence)
    # saved_model = r'C:\VMShare\videoclassification\data\checkpoints\lstm-features.070-0.135.hdf5'
    # predict2(sequence, saved_model, None, class_limit=10)


    # id = uuid.uuid4
    # temp_path = os.path.join(Config.root_temp, str(id))
    # if not os.path.exists(temp_path):
    #     os.makedirs(temp_path)


    predict_from_npy(
        r"C:\VMShare\videoclassification\data\sequences\40\v_ApplyEyeMakeup_g18_c04-40-features.npy",
        r'C:\VMShare\videoclassification\data\checkpoints\lstm-features.051-0.527.hdf5', 5
    )

    pass
