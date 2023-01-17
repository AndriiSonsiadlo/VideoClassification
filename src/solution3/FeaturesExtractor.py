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

        self.path = os.path.join(
            Config.root_data,
            'sequences',
            str(self.data.seq_length),
        )

    def extract(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Loop through data
        for video in tqdm(self.data.data, colour="green"):
            path = os.path.join(Config.root_img_seq_dataset, video[1], video[2])

            self.extract_for_one_video(path)

    def extract_for_one_video(self, video_dir):
        # Get the path to the sequence for this video
        *_, video_folder = split_path(video_dir)

        # Check if we already have it
        npy_filename = f'{video_folder}-{str(self.data.seq_length)}-features.npy'
        npy_file_path = os.path.join(self.path, npy_filename)
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
        np.save(npy_file_path, sequence)


def main():
    extractor = FeaturesExtractor()
    extractor.extract()
    # extractor.extract_for_one_video(r"C:\VMShare\videoclassification\data\test\BasketballDunk\v_BasketballDunk_g04_c03")


if __name__ == "__main__":
    main()
