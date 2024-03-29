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
import os.path

import numpy as np
from tqdm import tqdm

from src.solution3.config import Config
from src.solution3.data_process.extractor import Extractor
from src.solution3.objects.Dataset import Dataset


class FeaturesExtractor:
    cfg = Config()

    def __init__(self, seq_length=40, weights=None):
        # Set defaults
        self.seq_length = seq_length

        # Get the model
        self.model = Extractor(weights=weights)

    def extract(self):
        print("Extracting features...")

        class_folder_paths = glob.glob(os.path.join(self.cfg.root_img_seq_dataset, '*'))

        for class_folder in tqdm(class_folder_paths):
            video_folder_paths = glob.glob(os.path.join(class_folder, '*'))

            for video_folder in video_folder_paths:
                npy_folder_path = os.path.join(video_folder, str(self.seq_length))

                if not os.path.exists(npy_folder_path):
                    os.makedirs(npy_folder_path)

                self.extract_for_one_video(video_folder, npy_folder_path)

    def extract_for_one_video(self, video_dir, npy_path):
        # Get the path to the sequence for this video

        # Check if we already have it
        npy_file_path = os.path.join(npy_path, self.cfg.npy_filename)
        if os.path.isfile(npy_file_path):
            print(f"Exists: {npy_file_path}")
            return

        # Get the frames for this video
        frames_paths = Dataset.get_frames_for_sample(video_dir)

        # Now downsample to just the ones we need
        frames_paths = Dataset.rescale_list(frames_paths, self.seq_length)

        # Now loop through and extract features to build the sequence
        sequence = []
        for image_path in frames_paths:
            features = self.model.extract(image_path)
            sequence.append(features)

        # Save the sequence
        print(f"Saved features in path: {npy_file_path}")
        np.save(npy_file_path, sequence)
        return npy_file_path


def main():
    extractor = FeaturesExtractor()
    extractor.extract()
    # extractor.extract_for_one_video(r"C:\VMShare\videoclassification\data\test\BasketballDunk\v_BasketballDunk_g04_c03")


if __name__ == "__main__":
    main()
