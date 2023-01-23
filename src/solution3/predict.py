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
from src.solution3.objects.ModelData import ModelData
from src.solution3.utils import load_pickle_model


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


def predict_video(id, video_path, temp_path, model_data: ModelData):
    filename = "temp.avi"
    dest_path = os.path.join(temp_path, filename)

    FileMover.move_one_video(video_path, dest_path)
    FrameExtractor().extract_frames_for_one_video_prediction(dest_path, id)

    extractor = FeaturesExtractor(seq_length=model_data.seq_length)
    npy_path = extractor.extract_for_one_video(temp_path, temp_path)

    # Predict
    prediction = predict_from_npy(model_data, npy_path)
    print(prediction)
    pred_labeled = model_data.data.print_class_from_prediction(np.squeeze(prediction, axis=0))

    return pred_labeled
    # delete_temp_dir(temp_path)


def load_data_model(root_model):
    model_data_path = os.path.join(root_model, "data.pickle")
    model_data: ModelData = load_pickle_model(model_data_path)
    model_h5_path = os.path.join(root_model, "model")
    model_data.model = load_model(model_h5_path)
    print(model_data.data.get_classes())

    return model_data


def create_video_with_pred(temp_video_folder, label, score):
    src_path = os.path.join(temp_video_folder, "temp.avi")
    dest_path = os.path.join(temp_video_folder, "output.avi")

    video = cv2.VideoCapture(src_path)

    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False):
        print("Error reading video file")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)

    result = cv2.VideoWriter(dest_path,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             30, size)
    while True:
        ret, frame = video.read()
        if ret:
            output = frame.copy()
            text = f"{label}: {str(score)[:6]}"
            cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            result.write(output)
        else:
            break
    video.release()
    result.release()
    print(f"Output video was saved in: {dest_path}")


def main():
    # npy_path = r"C:\VMShare\videoclassification\data\img_seq_dataset\Basketball\v_Basketball_g02_c03\40\features.npy"
    # video_path = r"C:\Users\andrii\Downloads\pushups.mp4"
    video_path = r"D:\MAGISTERSKIE\Uczenie_glebokie\Action_Classification_CNN\videoclassification\data\videos\Archery\v_Archery_g11_c02.avi"
    root_model = r"D:\MAGISTERSKIE\Uczenie_glebokie\Action_Classification_CNN\videoclassification\data\models\1_reversed_lstm_10-classes_15-videos_2023-1-23_12-58"

    temp_id = uuid.uuid4()


    temp_path = create_temp_dir(temp_id)

    model_data = load_data_model(root_model)

    pred_labeled = predict_video(temp_id, video_path, temp_path, model_data)  # {"classlabel": 0.52, ...}
    label, score = next(iter(pred_labeled))  # classlabel, 0.52

    create_video_with_pred(temp_path, label, score)


if __name__ == '__main__':
    main()
