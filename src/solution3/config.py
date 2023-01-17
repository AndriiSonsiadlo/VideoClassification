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
    # root_temp = r"C:\VMShare\videoclassification\data\temp"
    root_img_seq_dataset = r"C:\VMShare\videoclassification\data\img_seq_dataset"

    data_file = os.path.join(root_data, 'data_file.csv') # .pickle
    video_type = "avi"

    ################################
    # prepare train and test lists #
    # and move videos to data dir  #
    ################################

    # method for choosing train and test videos
    method = "custom"

    # parameters for "ucf list" method
    version = "01"
    train_list_file = os.path.join(root_data, 'ucfTrainTestlist', 'trainlist' + version + '.txt')
    test_list_file = os.path.join(root_data, 'ucfTrainTestlist', 'testlist' + version + '.txt')

    # parameters for "custom" method
    class_list: tuple[str] = ()                 # if list is empty use class_number parameters else use class_list
    class_number: int | list[str] | None = None   # 20 # 30 # None - all classes
    shuffle_classes: bool = False
    video_number_per_class: int | None = 15     # None - all videos
    shuffle_videos: bool = True
    test_split: float = 0.3


    ################################
    #      features extraction     #
    ################################

    npy_filename = "features.npy"

    # Number of frames to extract features for them
    seq_length_extr = 40
    # Number of classes to extract. Can be 1-101 or None for all.
    class_limit_extr: int | None = None

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

    # for lstm, conv_3d, c3d models
    image_shape = (80, 80, 3)
