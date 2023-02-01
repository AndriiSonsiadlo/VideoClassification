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

from solution2.config import Config
from solution2.data_process.utils import split_path


class FileMover:
    @staticmethod
    def get_train_test_lists(test_file, train_file, root_dataset):
        """
        Using one of the train/test files (01, 02, 03 or 04), get the filename
        breakdowns we'll later use to move everything.
        """

        # Build the test list.
        with open(test_file) as fin:
            test_list = [os.path.join(root_dataset, row.strip()) for row in list(fin)]

        # Build the train list. Extra step to remove the class index.
        with open(train_file) as fin:
            train_list = [os.path.join(root_dataset, row.strip()) for row in list(fin)]
            train_list = [row.split(' ')[0] for row in train_list]

        # Set the groups in a dictionary.
        file_groups = {
            'train': train_list,
            'test': test_list
        }
        return file_groups

    @staticmethod
    def get_custom_list(root_dataset, class_number, shuffle_classes, video_number_per_class, shuffle_videos, test_split,
                        video_type):
        train_list = []
        test_list = []
        action_dirs = []

        dirs = os.listdir(root_dataset)
        random.shuffle(dirs) if shuffle_classes else ...
        for action_dir in dirs:
            action_dirs.append(action_dir)
            if len(action_dirs) == class_number:
                break

        for action_dir in tqdm(action_dirs):
            all_video_paths = glob.glob(f"{root_dataset}/{action_dir}/*.{video_type}")
            random.shuffle(all_video_paths) if shuffle_videos else ...

            video_paths = []
            for video_path in all_video_paths:
                video_paths.append(video_path)
                if len(video_paths) == video_number_per_class:
                    break

            test_number = math.ceil(len(video_paths) * test_split)
            train_list.extend(video_paths[test_number:])
            test_list.extend(video_paths[:test_number])

        file_groups = {
            'train': train_list,
            'test': test_list
        }
        return file_groups

    @staticmethod
    def move_files(file_groups, data_path):
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

                video_dir = os.path.join(data_path, group, classname, filename_no_ext)
                dest = os.path.join(data_path, group, classname, filename_no_ext, filename)

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


def prepare_lists(cfg=Config()):
    match cfg.method.lower():
        case "ucflist":
            return FileMover.get_train_test_lists(root_dataset=cfg.root_dataset,
                                                  test_file=cfg.test_list_file,
                                                  train_file=cfg.train_list_file)
        case "custom":
            return FileMover.get_custom_list(root_dataset=cfg.root_dataset,
                                             class_number=cfg.class_number,
                                             shuffle_classes=cfg.shuffle_classes,
                                             video_number_per_class=cfg.video_number_per_class,
                                             shuffle_videos=cfg.shuffle_videos,
                                             test_split=cfg.test_split,
                                             video_type=cfg.video_type)
        case other:
            raise ValueError("Given method is not correct, use one of: ['ucflist', 'custom']")


def main():
    """
    Prepare train and test lists according to parameters in Config
    """
    file_groups = prepare_lists()
    FileMover.move_files(file_groups, Config.root_data)


if __name__ == '__main__':
    main()
