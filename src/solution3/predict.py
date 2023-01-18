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

import cv2
import numpy as np
from keras.models import load_model

from src.solution3.config import Config
from src.solution3.data_process.FeaturesExtractor import FeaturesExtractor
from src.solution3.data_process.FileMover import FileMover
from src.solution3.data_process.FrameExtractor import FrameExtractor
from src.solution3.objects.ModelData import load_pickle_model, ModelData


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


def create_temp_dir(id, cfg=Config()):
    path = os.path.join(cfg.root_temp, str(id))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def delete_temp_dir(id, cfg=Config()):
    path = os.path.join(cfg.root_temp, str(id))
    if os.path.exists(path):
        os.remove(path)


def predict_from_npy(model_data, npy_path):
    # Get the data and process it.
    sample = np.load(npy_path)

    # Predict!
    prediction = model_data.model.predict(np.expand_dims(sample, axis=0))

    return prediction


def predict_video(video_path, model_data: ModelData):
    id = uuid.uuid4()

    temp_path = create_temp_dir(id)
    filename = "temp.avi"
    dest_path = os.path.join(temp_path, filename)

    FileMover.move_one_video(video_path, dest_path)
    FrameExtractor().extract_frames_for_one_video_prediction(dest_path, id)

    extractor = FeaturesExtractor(seq_length=model_data.seq_length)
    npy_path = extractor.extract_for_one_video(temp_path, temp_path)

    # Predict
    prediction = predict_from_npy(model_data, npy_path)
    print(prediction)
    model_data.data.print_class_from_prediction(np.squeeze(prediction, axis=0))

    # delete_temp_dir(temp_path)


def main():
    # npy_path = r"C:\VMShare\videoclassification\data\img_seq_dataset\Basketball\v_Basketball_g02_c03\40\features.npy"

    video_path = r"C:\VMShare\datasets\ucf-101\UCF-101\PommelHorse\v_PommelHorse_g08_c01.avi"
    root_model = r"C:\VMShare\videoclassification\data\models\10837b08-beba-4f16-8add-53e432eb9bcc-10classes-15videos"

    model_data_path = os.path.join(root_model, "data.pickle")
    model_data: ModelData = load_pickle_model(model_data_path)
    model_h5_path = os.path.join(root_model, "model")
    model_data.model = load_model(model_h5_path)
    print(model_data.data.get_classes())

    predict_video(video_path, model_data)


if __name__ == '__main__':
    main()
