"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
import os.path
import shutil

from tqdm import tqdm

from src.solution3.config import Config
from src.solution3.data_process.DatasetPreparator import DatasetPreparator
from src.solution3.utils import split_path


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
            dest_path = os.path.join(dataset_dir, classname, filename_no_ext, filename)

            # Check if we have already moved this file, or at least that it exists to move
            if not os.path.exists(video):
                print("Can't find %s to move. Skipping." % filename)
                continue

            # Check if this class exists
            if not os.path.exists(video_dir):
                print("Creating folder for %s" % video_dir)
                os.makedirs(video_dir)

            # Check if this class exists
            if os.path.exists(dest_path):
                print(f"Video exists: {dest_path}")
                continue
            FileMover.move_one_video(video, dest_path)
        print("Done.")

    @staticmethod
    def move_one_video(video_path, dest_path):
        # Move video
        print("Copying video to %s" % dest_path)
        shutil.copyfile(video_path, dest_path)

def main():
    """
    Prepare train and test lists according to parameters in Config
    """
    videos = DatasetPreparator.prepare_lists(class_number=1, video_number_per_class=3, shuffle_videos=False)
    FileMover.move_files(videos)


if __name__ == '__main__':
    main()
