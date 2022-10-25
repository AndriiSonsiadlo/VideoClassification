"""
This script devides each video from dataset videos into single frames.
"""
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='data_prep.log', encoding='utf-8', level=logging.DEBUG)

import glob
import os
import sys
import random

import cv2 as cv
import numpy as np
from tqdm import tqdm

VIDEOS_BASE_PATH = "data/videos/"
FRAMES_BASE_PATH = "data/frames/"

VIDEO_FILE_TYPE = 'avi'
FRAME_FILE_TYPE = 'png'

# Extract frames for actions
# sensitive to letter register
# an empty list means each actions
OPTIONAL_ACTIONS = ["HorseRiding"]
MAX_FRAMES_PER_ACTION = 8000
logger = logging.getLogger('data_prep')
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


def extract_frames_by_video_path(video_path: str, action: str = "unknown", frame_dir_save: str = None) -> None:
    # Parse video name and frame dir
    video_name, ext = os.path.splitext(os.path.basename(video_path))
    frame_dir = f"{FRAMES_BASE_PATH}/{action}/{video_name}" if not frame_dir_save else frame_dir_save

    # Check if frames dir exists
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)

    # Read video and save frames
    capture = cv.VideoCapture(video_path)
    while capture.isOpened():
        success, frame = capture.read()
        if success:
            nr_frame = int(capture.get(cv.CAP_PROP_POS_FRAMES))
            cv.imwrite(f"{frame_dir}/{nr_frame}.{FRAME_FILE_TYPE}", frame)
        else:
            break
    capture.release()


def extract_process() -> None:
    """ Call this function to extract video frames from all dataset or specified actions in OPTIONAL_ACTIONS param """

    # check if directory exists.
    if not os.path.isdir(VIDEOS_BASE_PATH):
        logger.error(f'NO VIDEO FILES TO EXTRACT IN "{VIDEOS_BASE_PATH}"')
        sys.exit(f"Missing video data in {VIDEOS_BASE_PATH}")

    action_dirs = os.listdir(VIDEOS_BASE_PATH)
    if OPTIONAL_ACTIONS:
        action_dirs = list(set(action_dirs) & set(OPTIONAL_ACTIONS))

    for action in tqdm(action_dirs, colour="green"):
        for video_path in tqdm(glob.glob(f"{VIDEOS_BASE_PATH}/{action}/*.{VIDEO_FILE_TYPE}")):
            extract_frames_by_video_path(video_path, action=action)


def fetch_frames_by_action(action_videos_dir_path: str, n_frames) -> list:
    frames_list = list()
    logger.debug('reading video directories')
    for video_dir in os.listdir(action_videos_dir_path):
        video_dir_path = os.path.join(action_videos_dir_path, video_dir)
        # logger.debug(f'reading video dir {video_dir_path}')
        for frame_file in random.sample(os.listdir(video_dir_path), n_frames):
            frame_path = os.path.join(video_dir_path, frame_file)
            frame = cv.imread(frame_path)
            if frame is not None:
                # resize step if needed
                # frame = cv.resize(frame, (image_height, image_width)) frames are width=320, height=240
                normalized_frame = frame / 255
                frames_list.append(normalized_frame)
            else:
                logger.debug("frame failed to read")

    return frames_list


def create_dataset(action_classes: list[str] = None, frames_per_video: int = None) -> (np.ndarray, np.ndarray):
    global OPTIONAL_ACTIONS

    if action_classes is not None:
        OPTIONAL_ACTIONS = action_classes

    features = list()
    labels = list()

    action_dirs = os.listdir(FRAMES_BASE_PATH)
    if OPTIONAL_ACTIONS:
        action_dirs = list(set(action_dirs) & set(OPTIONAL_ACTIONS))

    for action_index, action_name in enumerate(action_dirs):
        logger.debug(f'CREATING DATASET FOR {action_name}')
        frames_action_videos_dir = f'{FRAMES_BASE_PATH}/{action_name}'

        num_videos = len(os.listdir(frames_action_videos_dir))
        if frames_per_video is None:
            frames_per_video = MAX_FRAMES_PER_ACTION//num_videos

        action_frames_num = min(frames_per_video*num_videos, MAX_FRAMES_PER_ACTION)
        logger.debug(f'num frames per video: {frames_per_video}')
        logger.debug(f'action frames max: {action_frames_num}')
        action_frames = fetch_frames_by_action(frames_action_videos_dir, frames_per_video)

        # Adding randomly selected frames to the features list
        features.extend(random.sample(action_frames, action_frames_num))

        # Adding Fixed number of labels to the labels list
        labels.extend([action_index] * action_frames_num)
        logger.debug(f'completed frame collecting for action {action_name}')
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels



def run():

    extract_process()
    # data, labels = create_dataset(frames_per_video=10)
    # print(len(labels))


if __name__ == '__main__':
    run()
