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

from solution3.data_process.DatasetPreparator import prepare_lists
from solution3.config import Config
from solution3.data_process.utils import split_path


class FileMover:
    @staticmethod
    def move_files(file_groups, dataset_dir=Config.root_img_seq_dataset):
        """This assumes all of our files are currently in _this_ directory.
        So move them to the appropriate spot. Only needs to happen once.
        """

        # Do each of our groups
        for group, videos in tqdm(file_groups.items()):
            print("%" * 20, group, "%" * 20)
            # Do each of our videos
            for video_path in tqdm(videos):

                # Get the parts
                parts = split_path(video_path)
                classname = parts[-2]
                filename = parts[-1]
                filename_no_ext = filename.split(".")[0]

                video_dir = os.path.join(dataset_dir, classname, filename_no_ext)
                dest = os.path.join(dataset_dir, classname, filename_no_ext, filename)

                # Check if we have already moved this file, or at least that it exists to move
                if not os.path.exists(video_path):
                    print("Can't find %s to move. Skipping." % filename)
                    continue

                # # Check if video already exists in dest directory
                # if os.path.exists(video_path):
                #     print("Video already exist in destination %s. Skipping." % dest)
                #     continue

                # Check if this class exists
                if not os.path.exists(video_dir):
                    print("Creating folder for %s" % video_dir)
                    os.makedirs(video_dir)

                # Check if this class exists
                if os.path.exists(dest):
                    print(f"Video exists: {dest}")
                    continue

                # Move video
                print("Copying %s to %s" % (video_path, dest))
                shutil.copyfile(video_path, dest)

        print("Done.")


def main():
    """
    Prepare train and test lists according to parameters in Config
    """
    file_groups = prepare_lists()
    FileMover.move_files(file_groups, Config.root_img_seq_dataset)


if __name__ == '__main__':
    main()
