"""
This script devides each video from dataset videos into single frames.
"""
import logging
import glob
import os
import time

import cv2
from tqdm import tqdm
from definitions import ROOT_DIR

from src.utils import FRAMES_BASE_PATH, FRAME_FILE_TYPE, VIDEOS_BASE_PATH, VIDEO_FILE_TYPE
from src.utils.utils import create_dir

TESTING_VIDEOS = ["Basketball", "VolleyballSpiking", "PushUps"]
from src.utils.optical_flow import of_preprocessing, optical_flow_prep

logger = logging.getLogger(__name__)
def extract_frames_by_video_path(video_path: str, action: str = "unknown", frames_per_video=-1,
                                 frame_dir_save: str = None) -> None:
    # Parse video name and frame dir
    video_name, ext = os.path.splitext(os.path.basename(video_path))
    frame_dir = f"{FRAMES_BASE_PATH}/{action}/{video_name}" if not frame_dir_save else frame_dir_save

    # Check if frames dir exists
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)

    # Read video and save frames
    capture = cv2.VideoCapture(video_path)

    if frames_per_video > 0:
        video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indexes = [fr_index for fr_index in range(0, video_frame_count - 1, video_frame_count // frames_per_video)]
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if success:
                nr_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.imwrite(f"{frame_dir}/{nr_frame}.{FRAME_FILE_TYPE}", frame)
            else:
                break
    else:
        while capture.isOpened():
            success, frame = capture.read()
            if success:
                nr_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.imwrite(f"{frame_dir}/{nr_frame}.{FRAME_FILE_TYPE}", frame)
            else:
                break
    capture.release()


def extract_process(action_dirs: list = None, frames_per_video=-1) -> None:
    """
    Call this function to extract video frames from all dataset or specified actions in OPTIONAL_ACTIONS param
    :param action_dirs: Extract frames for specific actions. Sensitive to letter register
    :param frames_per_video: ...
    """

    if not action_dirs:
        action_dirs = os.listdir(VIDEOS_BASE_PATH)

    for action in tqdm(action_dirs, colour="green"):
        for video_path in tqdm(glob.glob(f"{VIDEOS_BASE_PATH}/{action}/*.{VIDEO_FILE_TYPE}")):
            extract_frames_by_video_path(video_path=video_path, action=action, frames_per_video=frames_per_video)


def generate_data(data_dir, UCF_dir, classes, train_v_test=0.7):
    start_time = time.time()
    sequence_length = 10
    image_size = (216, 216, 3)
    # logger.debug(f'data dir: {data_dir}')
    # logger.debug(f'ucf dir: {UCF_dir}')
    dest_dir = os.path.join(data_dir, 'UCF-Preprocessed-OF')

    #preprocessing for optical flow data
    of_preprocessing(UCF_dir, dest_dir, sequence_length, image_size, train_v_test , classes, overwrite=True, normalization=False,
                     mean_subtraction=False, continuous_seq=True)

    # compute optical flow data
    src_dir = os.path.join(ROOT_DIR, 'data/UCF-Preprocessed-OF')
    dest_dir = os.path.join(ROOT_DIR, 'data/OF_data')
    create_dir(dest_dir)

    optical_flow_prep(src_dir, dest_dir, mean_sub=True, overwrite=True)
    logger.info(f'Finished generating data in {int(time.time()-start_time / 60)} minutes')


# extract_process(TESTING_VIDEOS, 10)