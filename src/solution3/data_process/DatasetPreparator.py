import glob
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
    def get_custom_list(root_dataset: str, included_classes: list, class_number: int, shuffle_classes: bool,
                        video_number_per_class: int, shuffle_videos: bool, video_type: str):
        video_list = []
        action_dirs = []

        class_dirs = os.listdir(root_dataset)
        random.shuffle(class_dirs) if shuffle_classes else ...

        for incl_class in included_classes:
            if incl_class in class_dirs:
                video_list.append(incl_class)

        for action in class_dirs:
            if len(action_dirs) >= class_number:
                break
            action_dirs.append(action)

        for action_dir in tqdm(action_dirs):
            all_video_paths = glob.glob(f"{root_dataset}/{action_dir}/*.{video_type}")
            random.shuffle(all_video_paths) if shuffle_videos else ...

            video_paths = []
            for video_path in all_video_paths:
                video_paths.append(video_path)
                if len(video_paths) >= video_number_per_class:
                    break

            video_list.extend(video_paths)

        return video_list

    @staticmethod
    def prepare_lists(cfg=Config(), method="custom", included_classes: list = (), class_number=10,
                      shuffle_classes=False, video_number_per_class=5, shuffle_videos=True):
        match method.lower():
            case "ucflist":
                return DatasetPreparator.get_train_test_lists(test_file=cfg.test_list_file,
                                                              train_file=cfg.train_list_file)
            case "custom":
                return DatasetPreparator.get_custom_list(root_dataset=cfg.root_dataset,
                                                         included_classes=included_classes,
                                                         class_number=class_number,
                                                         shuffle_classes=shuffle_classes,
                                                         video_number_per_class=video_number_per_class,
                                                         shuffle_videos=shuffle_videos,
                                                         video_type=cfg.video_type)
            case other:
                raise ValueError("Given method is not correct, use one of: ['ucflist', 'custom']")


if __name__ == '__main__':
    DatasetPreparator.prepare_lists()
