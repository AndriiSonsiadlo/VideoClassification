import os
import re
from dataclasses import dataclass

from solution3.config import Config
from data_process.utils import split_path


@dataclass(init=True, unsafe_hash=True, frozen=True, order=True)
class ActionElement:
    video_folder_path: str
    class_name: str

    @property
    def video_file_path(self) -> str:
        video_filename = f"{split_path(self.video_folder_path)[-1]}.{Config.video_type}"
        path = os.path.join(self.video_folder_path, video_filename)
        return path

    @property
    def exists_video(self):
        return bool(os.path.exists(self.video_file_path))

    def exists_npy(self, seq_length):
        path = self.path_npy(seq_length)
        return bool(os.path.exists(path))

    def path_npy(self, seq_length):
        path = os.path.join(self.video_folder_path, str(seq_length), Config.npy_filename)
        return path

    @property
    def label(self) -> str:
        re_label = r"[A-Z]{1}[A-Za-z]+[a-z]+"
        label = re.search(re_label, self.filename)
        if label:
            label = label.group()
        else:
            label = "unknown"
            raise Exception("Cannot parse filename to label according to implemented ReGex pattern")
        return label

    @property
    def filename(self) -> str:
        return split_path(self.video_folder_path)[-1]

    def __str__(self):
        return self.filename
