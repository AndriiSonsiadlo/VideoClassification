import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    root_dataset: path to dataset videos
    available methods: ["ucflist", "custom"]
        ucflist: use defined ucfList for split train and test files
        random: choose random classes with parameters
        alphabetic: similar as random method, classes is chosen alphabetic

    """

    ################################
    #      Default parameters      #
    ################################

    root_dataset = r"C:\VMShare\datasets\ucf-101\UCF-101"
    root_data = r"C:\VMShare\videoclassification\data"
    root_img_seq_dataset = r"C:\VMShare\videoclassification\data\img_seq_dataset"
    root_models = r"C:\VMShare\videoclassification\models"

    data_file = os.path.join(root_data, 'data.pickle') # .pickle

    npy_filename = "features.npy"
    video_type = "avi"

    ################################
    # prepare train and test lists #
    # and move videos to data dir  #
    ################################

    # method for choosing train and test videos
    method = "custom"

    # parameters for "custom" method
    class_list: tuple[str] = ()                 # if list is empty use class_number parameters else use class_list
    test_split: float = 0.3

    ################################
    #      Learning parameters     #
    ################################

    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lstm'
    saved_model = None  # None or weights file
    class_limit_lrn = None  # int, can be 1-101 or None
    seq_length_lrn = 40
    load_to_memory = False  # preload the sequences into memory
    batch_size = 32
    nb_epoch = 100

