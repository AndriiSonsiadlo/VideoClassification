import glob
import math
import os
import random

from tqdm import tqdm

from src.solution3.config import Config


class DatasetPreparator:
    @staticmethod
    def get_train_test_lists(test_file, train_file, root_dataset):
        """
        Using one of the train/test files (01, 02, 03 or 04), get the filename
        breakdowns we'll later use to move everything.
        """

        video_list = []

        # Build the test list.
        with open(test_file) as fin:
            test_list = [os.path.join(root_dataset, row.strip()) for row in list(fin)]

        # Build the train list. Extra step to remove the class index.
        with open(train_file) as fin:
            train_list = [os.path.join(root_dataset, row.strip()) for row in list(fin)]
            train_list = [row.split(' ')[0] for row in train_list]

        video_list.extend(train_list)
        video_list.extend(test_list)
        return video_list

    @staticmethod
    def get_custom_list(root_dataset, class_number, shuffle_classes, video_number_per_class, shuffle_videos,
                        video_type):
        video_list = []
        action_dirs = []

        class_dirs = os.listdir(root_dataset)
        random.shuffle(class_dirs) if shuffle_classes else ...

        if Config.class_list:
            for class_name in Config.class_list:
                if class_name in class_dirs:
                    action_dirs.append(class_name)
                else:
                    print(f"{class_name} class does not exist in dataset. Skipping.")
        else:
            for action in class_dirs:
                if len(action_dirs) == class_number:
                    break
                action_dirs.append(action)

        for action_dir in tqdm(action_dirs):
            all_video_paths = glob.glob(f"{root_dataset}/{action_dir}/*.{video_type}")
            random.shuffle(all_video_paths) if shuffle_videos else ...

            video_paths = []
            for video_path in all_video_paths:
                video_paths.append(video_path)
                if len(video_paths) == video_number_per_class:
                    break

            video_list.extend(video_paths)

        return video_list

    @staticmethod
    def prepare_lists(cfg=Config(), method="custom", class_number=10, shuffle_classes=False, video_number_per_class=5, shuffle_videos=True):
        match method.lower():
            case "ucflist":
                return DatasetPreparator.get_train_test_lists(root_dataset=cfg.root_dataset,
                                                              test_file=cfg.test_list_file,
                                                              train_file=cfg.train_list_file)
            case "custom":
                return DatasetPreparator.get_custom_list(root_dataset=cfg.root_dataset,
                                                         class_number=class_number,
                                                         shuffle_classes=shuffle_classes,
                                                         video_number_per_class=video_number_per_class,
                                                         shuffle_videos=shuffle_videos,
                                                         video_type=cfg.video_type)
            case other:
                raise ValueError("Given method is not correct, use one of: ['ucflist', 'custom']")
