"""
This script devides each video from dataset videos into single frames.
"""

import glob
import os

import cv2
from tqdm import tqdm

VIDEOS_BASE_PATH = "../../data/videos/"
FRAMES_BASE_PATH = "../../data/frames/"

VIDEO_FILE_TYPE = 'avi'
FRAME_FILE_TYPE = 'png'

# Extract frames for actions
# sensitive to letter register
# an empty list means each actions
OPTIONAL_ACTIONS = ["Basketball"]


def extract_frames_by_video_path(video_path: str, action: str = "unknown", frame_dir_save: str = None) -> None:
    # Parse video name and frame dir
    video_name, ext = os.path.splitext(os.path.basename(video_path))
    frame_dir = f"{FRAMES_BASE_PATH}/{action}/{video_name}" if not frame_dir_save else frame_dir_save

    # Check if frames dir exists
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)

    # Read video and save frames
    capture = cv2.VideoCapture(video_path)
    while capture.isOpened():
        success, frame = capture.read()
        if success:
            nr_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.imwrite(f"{frame_dir}/{nr_frame}.{FRAME_FILE_TYPE}", frame)
        else:
            break
    capture.release()


def extract_process() -> None:
    """ Call this function to extract video frames from all dataset or specified actions in OPTIONAL_ACTIONS param """

    action_dirs = os.listdir(VIDEOS_BASE_PATH)
    if OPTIONAL_ACTIONS:
        action_dirs = list(set(action_dirs) & set(OPTIONAL_ACTIONS))

    for action in tqdm(action_dirs, colour="green"):
        for video_path in tqdm(glob.glob(f"{VIDEOS_BASE_PATH}/{action}/*.{VIDEO_FILE_TYPE}")):
            extract_frames_by_video_path(video_path, action=action)
