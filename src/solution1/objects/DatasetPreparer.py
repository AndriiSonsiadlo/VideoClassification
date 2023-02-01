import glob
import math
import os
import random

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from solution1.objects.Action import Action
from solution1.objects.ConfigReader import ConfigReader
from solution1.objects.Singleton import Singleton


class DatasetPreparer(metaclass=Singleton):

    def __init__(self):
        ds_params = ConfigReader()()["dataset"]
        self.dataset_path = ds_params.get("dataset_path")
        self.video_file_type = ds_params.get("video_file_type")
        self.video_class_number = ds_params.getint("video_class_number")
        self.class_number = ds_params.getint("class_number")
        self.test_split = ds_params.getfloat("test_dataset_in_percent")

    def prepare_dataset(self):
        action_objs = []
        action_dirs = []

        dirs = os.listdir(self.dataset_path)
        random.shuffle(dirs)
        for action_dir in dirs:
            action_dirs.append(action_dir)
            if len(action_dirs) == self.class_number:
                break
        # action_dirs = random.sample(action_dirs, self.class_number)

        for action in tqdm(action_dirs):
            all_video_paths = glob.glob(f"{self.dataset_path}/{action}/*.{self.video_file_type}")

            random.shuffle(all_video_paths)
            video_paths = []
            for video_path in all_video_paths:
                video_paths.append(video_path)
                if len(video_paths) == self.video_class_number:
                    break

            # if self.video_class_number > 0:
            #     video_paths = random.sample(video_paths, self.video_class_number)

            random.shuffle(video_paths)
            test_number = math.ceil(len(video_paths) * self.test_split)
            train_videos = video_paths[test_number:]
            test_videos = video_paths[:test_number]

            act = Action(
                action_name=action,
                action_path=f"{self.dataset_path}/{action}",
                video_number=self.video_class_number,
                test_split=self.test_split,
                train_video_paths=train_videos,
                test_video_paths=test_videos
            )
            action_objs.append(act)
        return action_objs

    @classmethod
    def actions_to_dfs(cls, action_objs: list[Action]) -> tuple[DataFrame, DataFrame]:
        train_action_data = []
        test_action_data = []
        for action in action_objs:
            for train_video_path in action.train_video_paths:
                train_action_data.append((action.action_name, "train", train_video_path.split("\\")[-1], train_video_path))
            for test_video_path in action.test_video_paths:
                test_action_data.append((action.action_name, "test", test_video_path.split("\\")[-1], test_video_path))

        train_df = pd.DataFrame(data=train_action_data, columns=["label", "step", "video_name", "video_path"])
        test_df = pd.DataFrame(data=test_action_data, columns=["label", "step", "video_name", "video_path"])

        return train_df, test_df

    @classmethod
    def save_actions_to_csv(cls, action_objs: list[Action]):
        train_df, test_df = cls.actions_to_dfs(action_objs)
        train_df.to_csv("data/train.csv")
        test_df.to_csv("data/test.csv")

    @classmethod
    def save_dfs_to_csv(cls, train_df: DataFrame, test_df: DataFrame):
        train_df.to_csv("data/train.csv")
        test_df.to_csv("data/test.csv")

    @classmethod
    def read_dfs_from_csv(cls) -> tuple[DataFrame, DataFrame]:
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        return train_df, test_df

    @classmethod
    def display_dataset(cls, action_objs: list[Action]):
        train_df, test_df = cls.actions_to_dfs(action_objs)

        print("Train dataset")
        pd.set_option('display.max_rows', train_df.shape[0] + 1)
        print(train_df)

        print("Test dataset")
        pd.set_option('display.max_rows', test_df.shape[0] + 1)
        print(test_df)


if __name__ == '__main__':
    dp = DatasetPreparer()
    action_objs = dp.prepare_dataset()
    DatasetPreparer.display_dataset(action_objs)
