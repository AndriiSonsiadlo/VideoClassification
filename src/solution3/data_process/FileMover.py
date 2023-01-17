"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import glob
import math
import os
import os.path
import random
import shutil

from tqdm import tqdm

from data_process.FileMover01 import prepare_lists
from solution3.config import Config
from solution3.data_process.utils import split_path


class FileMover:
    @staticmethod
    def move_files(videos, cfg=Config()):
        """This assumes all of our files are currently in _this_ directory.
        So move them to the appropriate spot. Only needs to happen once.
        """

        dataset_dir = cfg.root_img_seq_dataset

        # Do each of our videos
        for video in tqdm(videos):
            # Get the parts
            parts = split_path(video)
            classname = parts[-2]
            filename = parts[-1]
            filename_no_ext = filename.split(".")[0]

            video_dir = os.path.join(dataset_dir, classname, filename_no_ext)
            dest_dir = os.path.join(dataset_dir, classname, filename_no_ext, filename)

            # Check if we have already moved this file, or at least that it exists to move
            if not os.path.exists(video):
                print("Can't find %s to move. Skipping." % filename)
                continue

            # Check if this class exists
            if not os.path.exists(video_dir):
                print("Creating folder for %s" % video_dir)
                os.makedirs(video_dir)

            # Check if this class exists
            if os.path.exists(dest_dir):
                print(f"Video exists: {dest_dir}")
                continue

            # Move video
            print("Copying %s to %s" % (video, dest_dir))
            shutil.copyfile(video, dest_dir)

        print("Done.")


def main():
    """
    Prepare train and test lists according to parameters in Config
    """
    videos = prepare_lists()
    FileMover.move_files(videos)


if __name__ == '__main__':
    pass
