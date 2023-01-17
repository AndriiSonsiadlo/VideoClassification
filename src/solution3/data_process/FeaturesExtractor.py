"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import glob
import shutil

import numpy as np
import os.path
from solution3.Dataset import Dataset
from tqdm import tqdm

from solution3.config import Config
from solution3.data_process.utils import split_path
from solution3.extractor import Extractor


class FeaturesExtractor:

    def __init__(self):
        # Get the dataset
        self.data = Dataset(seq_length=Config.seq_length_extr, class_limit=Config.class_limit_extr)

        # get the model.
        self.model = Extractor()
        self.filename = "features.npy"

    @classmethod
    def generate_path(cls, video_path, seq_length):
        return os.path.join(Config.root_img_seq_dataset, video_path, seq_length)

    def extract(self):
        print("Extracting features")

        class_folders = glob.glob(os.path.join(Config.root_img_seq_dataset, '*'))
        for class_folder in tqdm(class_folders):
            video_folders = glob.glob(os.path.join(class_folder, '*'))
            for video_folder in video_folders:
                npy_folder_path = self.generate_path(video_folder, str(Config.seq_length_extr))
                if not os.path.exists(npy_folder_path):
                    os.makedirs(npy_folder_path)
                self.extract_for_one_video(video_folder, npy_folder_path)

    def extract_for_one_video(self, video_dir, npy_path):
        # Get the path to the sequence for this video
        *_, video_folder = split_path(video_dir)

        # Check if we already have it
        npy_filename = "features.npy"
        npy_file_path = os.path.join(npy_path, npy_filename)
        if os.path.isfile(npy_file_path):
            print(f"Exist: {npy_file_path}")
            return

        # Get the frames for this video
        frames = self.data.get_frames_for_sample(video_dir)

        # Now downsample to just the ones we need
        frames = self.data.rescale_list(frames, self.data.seq_length)

        # Now loop through and extract features to build the sequence
        sequence = []
        for image in frames:
            features = self.model.extract(image)
            sequence.append(features)

        # Save the sequence
        print(f"Saved features in path: {npy_file_path}")
        np.save(npy_file_path, sequence)


def refactor_npy():
    seq_path = os.path.join(Config.root_data, "sequences", "40")
    npy_files = glob.glob(os.path.join(seq_path, '*.npy'))
    npy_filename = "features.npy"

    for file in npy_files:
        npy_file_name = split_path(file)[-1]
        video_name_no_ext = npy_file_name.split("-")[0]
        video_class = video_name_no_ext.split("_")[1]

        dest_path = os.path.join(Config.root_img_seq_dataset, video_class, video_name_no_ext, "40")
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        npy_filename_path = os.path.join(dest_path, npy_filename)

        print(f"Moved: {file}")
        shutil.copyfile(file, npy_filename_path)


def main():
    extractor = FeaturesExtractor()
    extractor.extract()
    # extractor.extract_for_one_video(r"C:\VMShare\videoclassification\data\test\BasketballDunk\v_BasketballDunk_g04_c03")


if __name__ == "__main__":
    main()
