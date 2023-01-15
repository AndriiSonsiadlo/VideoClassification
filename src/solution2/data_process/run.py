import glob
import math
import os
import random
import shutil

from tqdm import tqdm

from solution2.data_process.FileMover01 import FileMover, prepare_lists
from solution2.config import Config


def main():
    file_groups = prepare_lists()
    FileMover.move_files(file_groups, Config.data_path)
    FileExtractor.extract_files()


if __name__ == '__main__':
    main()
